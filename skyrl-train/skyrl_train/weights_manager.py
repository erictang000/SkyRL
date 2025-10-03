from typing import List
from skyrl_train.workers.worker import PPORayActorGroup
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from ray import ObjectRef
import asyncio
from skyrl_train.utils import Timer
import ray


class ConditionalWeightsManager:
    def __init__(self, weights_manager, condition):
        self.weights_manager = weights_manager
        self.condition = condition

    def update_condition(self, condition):
        self.condition = condition

    def __enter__(self):
        if self.condition:
            self.weights_manager.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.condition:
            return self.weights_manager.__exit__(exc_type, exc_val, exc_tb)
        return False


class InferenceWeightsManager:
    """Manages weight syncing and offloading/backloading between the policy model and the InferenceEngines.

    This class is used to synchronize the weights of the policy model to the InferenceEngines.
    It also wakes up the inference engine if `colocate_all` is enabled.

    If `no_sync` is enabled, the weights will not be synchronized.
    Optionally puts the inference engine to sleep on exit if `sleep_on_exit` is `True`
    """

    def __init__(
        self,
        policy_model: PPORayActorGroup,
        inference_engine_client: InferenceEngineClient,
        colocate_all: bool,
        sleep_on_exit: bool = True,
        no_sync: bool = False,
    ):
        self.policy_model = policy_model
        self.inference_engine_client = inference_engine_client
        self.colocate_all = colocate_all
        self.sleep_on_exit = sleep_on_exit
        self.no_sync = no_sync

    def sync_policy_weights_to_inference_engines(self) -> List[ObjectRef]:
        return self.policy_model.async_run_ray_method(
            "pass_through", "broadcast_to_inference_engines", self.inference_engine_client
        )

    async def async_sync_policy_weights_to_inference_engines(self):
        return await self.policy_model.async_run_method(
            "pass_through", "broadcast_to_inference_engines", self.inference_engine_client
        )

    def __enter__(self):
        """Synchronous inference weights manager __enter__ method

        Syncs weights to InferenceEngines and wakes up the inference engine if `colocate_all` is enabled.

        If `colocate_all` is enabled, the policy model needs to be backloaded to GPU before
        calling this function. It will be offloaded to CPU after this method returns.

        We wake up the inference engine in two phases to minimize the peak GPU memory usage if
        `colocate_all` is enabled.
        """
        if self.colocate_all:
            asyncio.run(self.inference_engine_client.wake_up(tags=["weights"]))
            from skyrl_train.utils import print_mem
            memory = ray.get(self.policy_model.async_run_ray_method("pass_through", "get_cuda_memory"))
            memory = memory[0]
            print_mem("memory after wake up inf engine", memory)

        if not self.no_sync:
            with Timer("sync_weights_to_inference_engines"):
                ray.get(self.sync_policy_weights_to_inference_engines())

            from skyrl_train.utils import print_mem
            memory = ray.get(self.policy_model.async_run_ray_method("pass_through", "get_cuda_memory"))
            memory = memory[0]
            print_mem("memory after sync weights", memory)

        if self.colocate_all:
            with Timer("offload_policy_model_to_cpu"):
                self.policy_model.offload_to_cpu()
            from skyrl_train.utils import print_mem
            memory = ray.get(self.policy_model.async_run_ray_method("pass_through", "get_cuda_memory"))
            memory = memory[0]
            print_mem("memory after offload to cpu", memory)
            asyncio.run(self.inference_engine_client.wake_up(tags=["kv_cache"]))

            from skyrl_train.utils import print_mem
            memory = ray.get(self.policy_model.async_run_ray_method("pass_through", "get_cuda_memory"))
            memory = memory[0]
            print_mem("memory after wake up kv cache", memory)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Offloads the inference engine if `colocate_all` is enabled."""
        if self.colocate_all and self.sleep_on_exit:
            asyncio.run(self.inference_engine_client.sleep())

    async def __aenter__(self):
        """Asyncio-compatible __enter__ method

        Syncs weights to InferenceEngines and wakes up the inference engine if `colocate_all` is enabled.

        If `colocate_all` is enabled, the policy model needs to be backloaded to GPU before
        calling this function. It will be offloaded to CPU after this method returns.

        We wake up the inference engine in two phases to minimize the peak GPU memory usage if
        `colocate_all` is enabled.
        """
        if self.colocate_all:
            await self.inference_engine_client.wake_up(tags=["weights"])

        if not self.no_sync:
            with Timer("sync_weights_to_inference_engines"):
                await self.async_sync_policy_weights_to_inference_engines()

        if self.colocate_all:
            with Timer("offload_policy_model_to_cpu"):
                self.policy_model.offload_to_cpu()
            await self.inference_engine_client.wake_up(tags=["kv_cache"])

    async def __aexit__(self, exc_type, exc_value, traceback):
        """Offloads the inference engine if `colocate_all` is enabled."""
        if self.colocate_all and self.sleep_on_exit:
            await self.inference_engine_client.sleep()
