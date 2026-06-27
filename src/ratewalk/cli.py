"""RateWalk CLI: run the full pipeline and write a JSON report.

    python -m ratewalk.cli run [--config path] [--out runs/]

Pipeline: data -> states -> markov -> curve -> horizon select -> sim (+jumps)
-> risk analytics -> duration grid -> transition sensitivity -> credit overlay
-> optional hedge agents -> optional vol policy. Everything is config-driven;
the report is stamped with the config hash and seed for reproducibility.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Optional

import numpy as np

from . import config as cfgmod
from . import obs
from .analytics import (var_cvar, distribution_moments, fit_gmm,
                        duration_grid_search, transition_sensitivity)
from .credit import apply_credit_overlay, single_coupon_sensitivity
from .curve import fit_curve_model
from .data import load_macro
from .hedge import build_agents, run_agents
from .horizon import select_horizon
from .instruments.book import build_book
from .markov import estimate_chain, estimate_conditional, resample_dirichlet
from .markov.diagnostics import log_likelihood, persistence_baseline_ll, stationary_distribution
from .sim import build_jump_models, run_simulation
from .states import build_rate_states, build_cpi_states, build_joint_series
from .vol import compare_policies


def _build_context(cfg):
    md = load_macro(country=cfg.country, source=cfg.data.source,
                    start=cfg.data.start, cpi_vintage=cfg.data.cpi_vintage)
    rs, rspace = build_rate_states(md.policy_rate, mode=cfg.state.rate_mode,
                                   increment_grid_bps=cfg.state.increment_grid_bps,
                                   level_bin_width_bps=cfg.state.level_bin_width_bps)
    cs, cspace = build_cpi_states(md.cpi_yoy, bins_yoy=cfg.state.cpi_bins_yoy)
    joint = build_joint_series(rs, cs, alignment=cfg.state.alignment)
    cm = fit_curve_model(md.curve, md.policy_rate,
                         add_noise=cfg.curve.add_residual_noise,
                         noise_scale_bps=cfg.curve.residual_scale_bps)
    rate_model = estimate_chain(rs, rspace, estimation=cfg.markov.estimation,
                                half_life_years=cfg.markov.half_life_years,
                                prior=cfg.markov.prior, prior_strength=cfg.markov.prior_strength)
    cpi_model = estimate_chain(cs, cspace, estimation=cfg.markov.estimation,
                               half_life_years=cfg.markov.half_life_years)
    cond = estimate_conditional(joint, rspace, cspace,
                                estimation=cfg.markov.estimation,
                                half_life_years=cfg.markov.half_life_years)
    conditional_P = [cond[lab].P for lab in cspace.labels]
    start_rate = float(md.policy_rate["rate"].iloc[-1])
    neutral_rate = float(md.policy_rate["rate"].mean())   # OU anchor = historical mean
    jumps = build_jump_models(cfg.sim.jumps) if cfg.sim.jumps_enabled else []
    return dict(md=md, rs=rs, rspace=rspace, cs=cs, cspace=cspace, cm=cm,
                rate_model=rate_model, cpi_model=cpi_model, conditional_P=conditional_P,
                start_rate=start_rate, neutral_rate=neutral_rate, jumps=jumps)


def _runner(cfg, ctx, *, conditional: bool):
    """Return run(horizon, duration=None, P=None, n_paths=None) -> ann returns
    and the SimResult of the last call (for terminal prices)."""
    last = {}

    def run(horizon, duration=None, P=None, n_paths=None):
        inst = ctx_instrument(cfg, duration)
        book = build_book(inst, lambda t: ctx["cm"].yield_at(ctx["start_rate"], t),
                          country=cfg.country)
        c2 = cfg
        if n_paths is not None:
            c2 = dataclasses.replace(cfg, sim=dataclasses.replace(cfg.sim, n_paths=n_paths))
        rate_P = P if P is not None else ctx["rate_model"].P
        res = run_simulation(
            c2, curve_model=ctx["cm"], book=book, rate_space=ctx["rspace"],
            rate_P=rate_P, start_rate=ctx["start_rate"], horizon_years=horizon,
            jump_models=ctx["jumps"], neutral_rate=ctx["neutral_rate"],
            cpi_P=(ctx["cpi_model"].P if conditional else None),
            conditional_P=(ctx["conditional_P"] if conditional else None),
            rng=np.random.default_rng(cfg.sim.seed))
        last["res"] = res
        return res.annualized_return

    return run, last


def ctx_instrument(cfg, duration):
    if duration is None:
        return cfg.instrument
    return dataclasses.replace(cfg.instrument, held_tenor_years=float(duration),
                               kind="single_bond")


def _build_viz(res, ann_pct, P, rspace, gmm) -> dict:
    """Compact, UI-ready visualization payloads: a fan chart of wealth and
    rate paths, the full transition matrix, and a return histogram with a GMM
    density overlay. Downsampled so the JSON stays small."""
    spy = res.steps_per_year
    n_steps = res.wealth_paths.shape[1]
    # monthly steps -> sample ~40 points along the horizon for the fan chart
    idx = np.unique(np.linspace(0, n_steps - 1, min(40, n_steps)).astype(int))
    pcts = [5, 25, 50, 75, 95]
    fan = {"t_years": (idx / spy).round(3).tolist(),
           "wealth": {f"p{p}": np.percentile(res.wealth_paths[:, idx], p, axis=0).round(3).tolist()
                      for p in pcts},
           "rate": {f"p{p}": np.percentile(res.rate_paths[:, idx], p, axis=0).round(3).tolist()
                    for p in pcts}}
    # return histogram
    counts, edges = np.histogram(ann_pct, bins=40)
    centers = ((edges[:-1] + edges[1:]) / 2).round(3)
    hist = {"centers": centers.tolist(), "counts": counts.tolist()}
    # GMM density overlay (if a mixture was fit)
    gmm_curve = None
    if gmm and gmm.get("n_components"):
        xs = np.linspace(float(ann_pct.min()), float(ann_pct.max()), 200)
        dens = np.zeros_like(xs)
        for w, m, s in zip(gmm["weights"], gmm["means"], gmm["stds"]):
            # means/stds are on the fraction scale; ann_pct is in percent
            mp, sp = m * 100.0, s * 100.0
            dens += w * np.exp(-0.5 * ((xs - mp) / sp) ** 2) / (sp * np.sqrt(2 * np.pi))
        gmm_curve = {"x": xs.round(3).tolist(), "density": dens.round(5).tolist()}
    return {
        "fan_chart": fan,
        "transition_matrix": {"labels": rspace.labels, "P": np.round(P, 4).tolist()},
        "return_histogram": hist,
        "gmm_density": gmm_curve,
    }


def run_pipeline(cfg) -> dict:
    obs.event(channel="run", kind="pipeline.start", country=cfg.country,
              config_hash=cfg.content_hash())
    ctx = _build_context(cfg)
    rspace = ctx["rspace"]

    # Markov diagnostics
    chain_ll = log_likelihood(ctx["rate_model"].P, ctx["rs"]["state"], rspace)
    base_ll = persistence_baseline_ll(ctx["rs"]["state"], rspace)
    stat = stationary_distribution(ctx["rate_model"].P)

    run_cond, last_cond = _runner(cfg, ctx, conditional=(cfg.markov.model == "conditional"))
    run_uncond, _ = _runner(cfg, ctx, conditional=False)

    # 1) horizon selection
    hz = select_horizon(cfg.horizon, lambda h: run_uncond(h, n_paths=min(cfg.sim.n_paths, 2000)))
    horizon = hz["selected"]

    # 2) headline simulation at the selected horizon
    ann = run_cond(horizon)
    res = last_cond["res"]
    ann_pct = ann * 100.0

    # 3) risk report
    moments = distribution_moments(ann)
    gmm = fit_gmm(ann, max_components=cfg.analytics.gmm_max_components) \
        if not moments["is_gaussian_5pct"] else {"note": "approximately gaussian; GMM skipped",
                                                  "is_gaussian": True}
    tail = {f"{int(l*100)}": var_cvar(ann, l) for l in cfg.analytics.var_levels}

    # 4) duration grid search
    dgrid = duration_grid_search(
        lambda d: run_uncond(horizon, duration=d, n_paths=min(cfg.sim.n_paths, 2000)),
        list(cfg.analytics.duration_grid_years), cfg.analytics.objective,
        crra_gamma=cfg.analytics.crra_gamma)

    # 5) transition sensitivity (Dirichlet draws); modest path count for speed
    n_draws = min(cfg.analytics.sensitivity_draws, 120)
    draws = resample_dirichlet(ctx["rate_model"], n_draws, np.random.default_rng(cfg.sim.seed + 1))
    sens = transition_sensitivity(
        lambda P: run_uncond(horizon, P=P, n_paths=800), draws)

    # 6) credit overlay + single-coupon sensitivity
    coupon = build_book(cfg.instrument, lambda t: ctx["cm"].yield_at(ctx["start_rate"], t),
                        country=cfg.country).holdings[0][1].coupon_rate
    credit = apply_credit_overlay(res, cfg.credit, horizon, coupon,
                                  np.random.default_rng(cfg.sim.seed + 2))
    coupon_sens = single_coupon_sensitivity(res, coupon, horizon, which_coupon=1)

    viz = _build_viz(res, ann_pct, ctx["rate_model"].P, rspace, gmm)

    report = {
        "config_hash": cfg.content_hash(),
        "country": cfg.country,
        "seed": cfg.sim.seed,
        "n_paths": cfg.sim.n_paths,
        "start_short_rate": round(ctx["start_rate"], 4),
        "data_source": ctx["md"].source,
        "markov": {
            "model": cfg.markov.model,
            "rate_states": rspace.labels,
            "stationary_distribution": dict(zip(rspace.labels, np.round(stat, 4).tolist())),
            "chain_log_likelihood": round(chain_ll, 4),
            "baseline_log_likelihood": round(base_ll, 4),
            "beats_baseline": bool(chain_ll > base_ll),
        },
        "horizon_selection": hz,
        "headline": {
            "horizon_years": horizon,
            "annualized_return_pct": {
                "mean": round(float(ann_pct.mean()), 4),
                "std": round(float(ann_pct.std()), 4),
                "p5": round(float(np.percentile(ann_pct, 5)), 4),
                "p50": round(float(np.percentile(ann_pct, 50)), 4),
                "p95": round(float(np.percentile(ann_pct, 95)), 4),
            },
            "init_investment": round(res.init_investment, 4),
        },
        "risk": {"VaR_CVaR": tail, "moments": moments, "gmm": gmm},
        "duration_grid": dgrid,
        "transition_sensitivity": sens,
        "viz": viz,
        "credit": {
            "enabled": cfg.credit.enabled,
            "n_defaults": credit.n_defaults,
            "mean_loss_vs_riskfree": round(credit.loss_vs_riskfree, 4),
            "single_coupon_miss_impact": round(coupon_sens, 4),
        },
    }

    # 7) optional hedging
    if cfg.hedge.enabled:
        agents = build_agents(cfg.hedge)
        report["hedge"] = run_agents(agents, res.terminal_wealth, res.horizon_return,
                                     cfg.hedge, horizon)

    # 8) optional vol allocation
    if cfg.vol.enabled:
        # signal confidence from the tightness of the sensitivity band
        band = sens["mean_return"]
        width = max(band["p95"] - band["p5"], 1e-9)
        conf = float(np.clip(1.0 - width / (abs(band["mean"]) + 1e-9), 0.0, 1.0))
        report["vol"] = compare_policies(
            ann, target_annual_vol=cfg.vol.target_annual_vol, horizon_years=horizon,
            signal_confidence=conf, confidence_scaling=cfg.vol.confidence_scaling)

    obs.event(channel="run", kind="pipeline.done", horizon=horizon,
              mean_ann_return_pct=report["headline"]["annualized_return_pct"]["mean"])
    return report


def run_walkforward(cfg, *, min_train: int = 120) -> dict:
    """Walk-forward validation: out-of-sample FOMC-decision forecasting (with a
    live nowcast) and a duration-timing backtest vs benchmarks. Everything is
    no-look-ahead: every prediction at month t uses only data public before t."""
    import numpy as np
    from .walkforward import prepare_series, compare_models, nowcast, duration_backtest
    from .walkforward.forecast import tau_sweep
    obs.event(channel="run", kind="walkforward.start", country=cfg.country)
    md = load_macro(country=cfg.country, source=cfg.data.source,
                    start=cfg.data.start, cpi_vintage=cfg.data.cpi_vintage)
    s = prepare_series(md, cfg, decision_freq=cfg.state.decision_freq)
    rng = np.random.default_rng(cfg.sim.seed)
    # tau fixed a priori at 50 (mild "~50 pseudo-observations" shrinkage), NOT
    # tuned on the test set. The sweep below shows the result is a broad plateau,
    # not a knife-edge, so the choice of tau is not driving the finding.
    SHRINK_TAU = 50.0
    forecast_eval = compare_models(s, min_train=min_train, n_dirichlet=200,
                                   shrink_tau=SHRINK_TAU, rng=rng)
    sweep = tau_sweep(s, [0, 5, 10, 20, 50, 100, 200, 500], min_train=min_train, rng=rng)
    live = nowcast(s, model="conditional_shrunk", shrink_tau=SHRINK_TAU, rng=rng)
    backtest = duration_backtest(s, md.curve, min_train=min_train)
    return {
        "config_hash": cfg.content_hash(),
        "country": cfg.country,
        "data_source": md.source,
        "min_train_months": min_train,
        "forecast_validation": forecast_eval,
        "tau_sweep": sweep,
        "nowcast": live,
        "duration_backtest": backtest,
    }


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(prog="ratewalk")
    sub = ap.add_subparsers(dest="cmd")
    pr = sub.add_parser("run", help="run the full pipeline")
    pr.add_argument("--config", type=str, default=None)
    pr.add_argument("--out", type=str, default="runs")
    wf = sub.add_parser("walkforward", help="out-of-sample forecast + backtest")
    wf.add_argument("--config", type=str, default=None)
    wf.add_argument("--out", type=str, default="runs")
    wf.add_argument("--min-train", type=int, default=120)
    args = ap.parse_args(argv)

    if args.cmd not in ("run", "walkforward"):
        ap.print_help()
        return 1

    cfg = cfgmod.load(Path(args.config) if args.config else None)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    obs.configure(log_dir=out_dir / "logs")

    if args.cmd == "walkforward":
        report = run_walkforward(cfg, min_train=args.min_train)
        out_path = out_dir / f"walkforward-{cfg.country}-{cfg.content_hash()}.json"
        with open(out_path, "w") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"wrote {out_path}")
        sm = report["forecast_validation"]["summary"]
        print(f"  forecast ({sm['eval_points']} OOS decisions, {cfg.state.decision_freq}) log-loss:")
        print(f"    climatology {sm['logloss_climatology']} | uncond {sm['logloss_unconditional']} | "
              f"cond-raw {sm['logloss_conditional_raw']} | shrunk(50) {sm['logloss_conditional_shrunk']} | "
              f"EB-tau {sm['logloss_conditional_eb']}")
        print(f"    raw CPI helps={sm['raw_cpi_conditioning_helps']}, "
              f"shrunk helps={sm['shrunk_cpi_conditioning_helps']}, EB helps={sm['eb_cpi_conditioning_helps']}")
        bt = report["duration_backtest"]
        print(f"  duration backtest: strategy Sharpe={bt['strategy']['sharpe']} vs "
              + ", ".join(f"{k} {v['sharpe']}" for k, v in bt['benchmarks'].items()))
        nc = report["nowcast"]
        print(f"  nowcast {nc['as_of']}: E[next move]={nc['expected_move_bps']}bps, "
              f"P(hold)={[d['prob'] for d in nc['distribution'] if d['move']=='+0bps'][0]}")
        return 0

    report = run_pipeline(cfg)
    out_path = out_dir / f"report-{cfg.content_hash()}.json"
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"wrote {out_path}")
    h = report["headline"]
    print(f"  horizon={h['horizon_years']}y  mean ann return="
          f"{h['annualized_return_pct']['mean']}%  "
          f"VaR95={report['risk']['VaR_CVaR']['95']['VaR']:.4f}  "
          f"best duration={report['duration_grid']['best']['duration']}y")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
