"""Tests for the experiment-only ER-1 V2 parameter search layer."""

import ast
from dataclasses import asdict
from pathlib import Path

import pytest

from epistemic_repair.beliefs.stochastic_likelihoods import (
    StochasticLikelihoodModel,
)
from epistemic_repair.er1.config import ER1_DEFAULT_CONFIG
from epistemic_repair.evaluation.er1_parameter_search import (
    CandidateStochasticLikelihoodModel,
    ER1CandidateParameters,
    analyze_initial_anomaly,
    build_candidate_scorecard,
    check_candidate_identifiability,
    rank_scorecards,
    run_candidate_episodes,
    run_parameter_search,
)
from epistemic_repair.failures.modes import FailureMode


def test_candidate_parameters_do_not_mutate_active_er1_config() -> None:
    before = asdict(ER1_DEFAULT_CONFIG)
    parameters = ER1CandidateParameters(
        normal_process_accuracy=0.75,
        shift_process_accuracy=0.85,
        sensor_normal_accuracy=0.95,
        corrupted_sensor_inversion_accuracy=0.85,
        latent_process_accuracy=0.85,
    )
    run_candidate_episodes(
        parameters,
        seed_count=2,
        budgets=(5, 8),
        thresholds=(0.90,),
    )
    assert asdict(ER1_DEFAULT_CONFIG) == before


def test_candidate_anomaly_probability_and_conditioning_math() -> None:
    parameters = ER1CandidateParameters(
        normal_process_accuracy=0.80,
        shift_process_accuracy=0.90,
        sensor_normal_accuracy=0.95,
        corrupted_sensor_inversion_accuracy=0.90,
        latent_process_accuracy=0.90,
    )
    analysis = analyze_initial_anomaly(parameters)
    expected_no_change = 0.80 * 0.05 + 0.20 * 0.95
    denominator = expected_no_change + 0.86 + 0.82 + 0.86
    assert analysis.no_change_likelihood == pytest.approx(0.23)
    assert analysis.no_change_posterior == pytest.approx(
        expected_no_change / denominator
    )
    assert sum(
        (
            analysis.no_change_posterior,
            analysis.world_shift_posterior,
            analysis.sensor_corruption_posterior,
            analysis.missing_latent_posterior,
        )
    ) == pytest.approx(1.0)


def test_v1_baseline_reproduces_active_probability_math() -> None:
    baseline = ER1CandidateParameters.v1_baseline()
    candidate = CandidateStochasticLikelihoodModel(baseline)
    active = StochasticLikelihoodModel()
    analysis = analyze_initial_anomaly(baseline)
    assert baseline.candidate_id == "v1_baseline"
    for hypothesis in (
        FailureMode.NO_STRUCTURAL_CHANGE,
        FailureMode.WORLD_SHIFT,
        FailureMode.SENSOR_CORRUPTION,
        FailureMode.MISSING_LATENT_VARIABLE,
    ):
        assert candidate.initial_anomaly_likelihood(hypothesis) == pytest.approx(
            active.initial_anomaly_likelihood(hypothesis)
        )
    assert analysis.no_change_likelihood == pytest.approx(0.14)
    assert analysis.no_change_posterior == pytest.approx(0.05223880597014926)


def test_candidate_search_episodes_are_reproducible() -> None:
    parameters = ER1CandidateParameters.v1_baseline()
    kwargs = {
        "seed_count": 4,
        "budgets": (5, 8),
        "thresholds": (0.90,),
    }
    assert run_candidate_episodes(parameters, **kwargs) == run_candidate_episodes(
        parameters, **kwargs
    )


def test_candidate_ranking_is_deterministic() -> None:
    first_parameters = ER1CandidateParameters(
        normal_process_accuracy=0.80,
        shift_process_accuracy=0.90,
        sensor_normal_accuracy=0.95,
        corrupted_sensor_inversion_accuracy=0.90,
        latent_process_accuracy=0.90,
    )
    second_parameters = ER1CandidateParameters.v1_baseline()
    cards = []
    for parameters in (first_parameters, second_parameters):
        episodes = run_candidate_episodes(
            parameters,
            seed_count=5,
            budgets=(5, 8),
            thresholds=(0.90,),
        )
        cards.append(build_candidate_scorecard(parameters, episodes))
    forward = rank_scorecards(cards)
    backward = rank_scorecards(reversed(cards))
    assert forward == backward


def test_recommendation_uses_best_finalist_even_when_it_is_baseline() -> None:
    study = run_parameter_search(phase1_seed_count=2, phase2_seed_count=2)
    assert study.recommended_candidate_id == rank_scorecards(study.finalists)[0].candidate_id


def test_identifiability_checks_flag_context_contrast_mismatch() -> None:
    preserved = check_candidate_identifiability(
        ER1CandidateParameters(
            normal_process_accuracy=0.80,
            shift_process_accuracy=0.85,
            sensor_normal_accuracy=0.95,
            corrupted_sensor_inversion_accuracy=0.85,
            latent_process_accuracy=0.85,
        )
    )
    weakened = check_candidate_identifiability(
        ER1CandidateParameters(
            normal_process_accuracy=0.80,
            shift_process_accuracy=0.85,
            sensor_normal_accuracy=0.95,
            corrupted_sensor_inversion_accuracy=0.85,
            latent_process_accuracy=0.90,
        )
    )
    assert preserved.passes
    assert preserved.repeat_trial_positive_information_gain
    assert not weakened.passes
    assert "context B" in weakened.notes


def test_active_v1_likelihood_behavior_remains_unchanged() -> None:
    model = StochasticLikelihoodModel()
    before = tuple(
        model.initial_anomaly_likelihood(hypothesis)
        for hypothesis in (
            FailureMode.NO_STRUCTURAL_CHANGE,
            FailureMode.WORLD_SHIFT,
            FailureMode.SENSOR_CORRUPTION,
            FailureMode.MISSING_LATENT_VARIABLE,
        )
    )
    CandidateStochasticLikelihoodModel(
        ER1CandidateParameters(
            normal_process_accuracy=0.75,
            shift_process_accuracy=0.85,
            sensor_normal_accuracy=0.95,
            corrupted_sensor_inversion_accuracy=0.85,
            latent_process_accuracy=0.85,
        )
    )
    after = tuple(
        model.initial_anomaly_likelihood(hypothesis)
        for hypothesis in (
            FailureMode.NO_STRUCTURAL_CHANGE,
            FailureMode.WORLD_SHIFT,
            FailureMode.SENSOR_CORRUPTION,
            FailureMode.MISSING_LATENT_VARIABLE,
        )
    )
    assert before == after == pytest.approx((0.14, 0.86, 0.82, 0.86))


def test_parameter_search_path_has_no_llm_or_provider_imports() -> None:
    project = Path(__file__).parents[1]
    sources = (
        project / "scripts" / "search_er1_parameters.py",
        project / "epistemic_repair" / "evaluation" / "er1_parameter_search.py",
        project / "epistemic_repair" / "evaluation" / "er1_calibration.py",
    )
    imported_modules = []
    for source in sources:
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)
    assert not any(
        token in module.lower()
        for module in imported_modules
        for token in ("llm", "gemini", "provider")
    )
