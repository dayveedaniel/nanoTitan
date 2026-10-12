import logging

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)


class PipelineComms:
    def __init__(
        self,
        prev_pp_rank: int,
        pp_group: ProcessGroup,
        next_pp_rank: int,
    ):
        self.prev_pp_rank = prev_pp_rank
        self.next_pp_rank = next_pp_rank
        self.pp_group = pp_group

    def recv_forward(self, microbatch_id: int, stage_input: torch.Tensor) -> torch.Tensor:
        logger.debug("Receiving forward microbatch %s", microbatch_id)
        dist.recv(stage_input, src=self.prev_pp_rank, group=self.pp_group)
        return stage_input

    def send_forward(self, microbatch_id: int, stage_output: torch.Tensor):
        logger.debug("Sending forward microbatch %s", microbatch_id)
        dist.send(stage_output, dst=self.next_pp_rank, group=self.pp_group)

    def recv_backward(self, microbatch_id: int, output_grad: torch.Tensor):
        logger.debug("Receiving backward microbatch %s", microbatch_id)
        dist.recv(output_grad, src=self.next_pp_rank, group=self.pp_group)
        return output_grad

    def send_backward(self, microbatch_id: int, input_grad: torch.Tensor):
        logger.debug("Sending backward microbatch %s", microbatch_id)
        dist.send(input_grad, dst=self.prev_pp_rank, group=self.pp_group)
