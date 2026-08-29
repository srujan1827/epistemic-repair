"""Provider-neutral LLM policies for ER-1 V2."""

from epistemic_repair.beliefs.stochastic_state import StochasticHypothesisBeliefs
from epistemic_repair.er1_v2.views import ER1V2BenchmarkAgentView
from epistemic_repair.llm.base import LLMClient
from epistemic_repair.llm.config import LLMConfig
from epistemic_repair.llm.schemas import (
    er1_full_autonomous_json_schema,
    er1_planner_only_json_schema,
    parse_er1_full_autonomous_response,
    parse_er1_planner_only_response,
)
from epistemic_repair.policies.llm import LLMPolicyResult, _generate_with_retries
from epistemic_repair.prompts.binary_er1_v2 import (
    BINARY_ER1_V2_PROMPT_VERSION,
    BINARY_ER1_V2_THRESHOLD_AWARE_PROMPT_VERSION,
    build_er1_v2_full_autonomous_prompt,
    build_er1_v2_planner_only_prompt,
    build_er1_v2_threshold_aware_autonomous_prompt,
)


class ER1V2FullAutonomousLLMPolicy:
    """V2 model responsible for beliefs, experiment choice, and stopping."""

    def __init__(self, client: LLMClient, config: LLMConfig) -> None:
        self._client = client
        self.config = config
        self.prompt_version = BINARY_ER1_V2_PROMPT_VERSION

    def decide(self, view: ER1V2BenchmarkAgentView) -> LLMPolicyResult:
        if type(view) is not ER1V2BenchmarkAgentView:
            raise TypeError(
                "ER-1 V2 autonomous policy requires ER1V2BenchmarkAgentView"
            )
        return _generate_with_retries(
            client=self._client,
            config=self.config,
            prompt=build_er1_v2_full_autonomous_prompt(view),
            schema=er1_full_autonomous_json_schema(),
            parser=parse_er1_full_autonomous_response,
            prompt_version=BINARY_ER1_V2_PROMPT_VERSION,
        )


class ER1V2ThresholdAwareAutonomousLLMPolicy:
    """Autonomous V2 policy told only its configured stopping threshold."""

    def __init__(
        self,
        client: LLMClient,
        config: LLMConfig,
        diagnosis_threshold: float,
    ) -> None:
        if (
            not isinstance(diagnosis_threshold, (int, float))
            or isinstance(diagnosis_threshold, bool)
            or not 0.0 < diagnosis_threshold <= 1.0
        ):
            raise ValueError("diagnosis_threshold must be in (0, 1]")
        self._client = client
        self.config = config
        self.diagnosis_threshold = float(diagnosis_threshold)
        self.prompt_version = BINARY_ER1_V2_THRESHOLD_AWARE_PROMPT_VERSION

    def decide(self, view: ER1V2BenchmarkAgentView) -> LLMPolicyResult:
        if type(view) is not ER1V2BenchmarkAgentView:
            raise TypeError(
                "ER-1 V2 threshold-aware policy requires "
                "ER1V2BenchmarkAgentView"
            )
        return _generate_with_retries(
            client=self._client,
            config=self.config,
            prompt=build_er1_v2_threshold_aware_autonomous_prompt(
                view,
                self.diagnosis_threshold,
            ),
            schema=er1_full_autonomous_json_schema(),
            parser=parse_er1_full_autonomous_response,
            prompt_version=self.prompt_version,
        )


class ER1V2PlannerOnlyLLMPolicy:
    """V2 experiment planner using benchmark-supplied normative beliefs."""

    def __init__(self, client: LLMClient, config: LLMConfig) -> None:
        self._client = client
        self.config = config
        self.prompt_version = BINARY_ER1_V2_PROMPT_VERSION

    def decide(
        self,
        view: ER1V2BenchmarkAgentView,
        normative_beliefs: StochasticHypothesisBeliefs,
    ) -> LLMPolicyResult:
        if type(view) is not ER1V2BenchmarkAgentView:
            raise TypeError(
                "ER-1 V2 planner policy requires ER1V2BenchmarkAgentView"
            )
        return _generate_with_retries(
            client=self._client,
            config=self.config,
            prompt=build_er1_v2_planner_only_prompt(view, normative_beliefs),
            schema=er1_planner_only_json_schema(),
            parser=parse_er1_planner_only_response,
            prompt_version=BINARY_ER1_V2_PROMPT_VERSION,
        )
