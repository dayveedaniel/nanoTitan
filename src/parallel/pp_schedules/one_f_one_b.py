from torch.profiler import record_function


def run_1F1B(pipeline, model, microbatch_x, microbatch_y):
    num_microbatches = len(microbatch_x)
    pp_group_size = pipeline.dim.pp_size
    pp_rank = pipeline.dim.pp_rank
    warmup_steps = min(num_microbatches, (pp_group_size - pp_rank))
    fwd_idx, bwd_idx = 0, 0

    with record_function("1f1b/warmup"):
        while fwd_idx < warmup_steps:
            if pipeline.dim.is_pp_first_stage:
                stage_input = microbatch_x[fwd_idx].to(pipeline.device)
            else:
                fwd_buffer = pipeline._activation_buffer()
                stage_input = pipeline.comms.recv_forward(fwd_idx, fwd_buffer)
            stage_output = pipeline.forward_microbatch(
                fwd_idx, model, stage_input, microbatch_y[fwd_idx]
            )
            if not pipeline.dim.is_pp_last_stage and fwd_idx < warmup_steps - 1:
                pipeline.comms.send_forward(fwd_idx, stage_output)
            fwd_idx += 1

    if fwd_idx == num_microbatches:
        pipeline.record_forward_completion()

    with record_function("1f1b/steady"):
        while True:
            # send fwd and recv bwd is skipped for the last stage
            if pipeline.dim.is_pp_last_stage:
                output_grad = None
            else:
                output_grad_buffer = pipeline._activation_buffer()
                output_grad = pipeline.comms.send_forward_recv_backward(
                    stage_output, output_grad_buffer
                )

            input_grad = pipeline.backward_microbatch(
                bwd_idx,
                output_grad,
                sync_gradients=bwd_idx == num_microbatches - 1,
            )
            bwd_idx += 1

            if fwd_idx == num_microbatches:
                if not pipeline.dim.is_pp_first_stage:
                    pipeline.comms.send_backward(bwd_idx - 1, input_grad)
                break

            if pipeline.dim.is_pp_first_stage:
                stage_input = microbatch_x[fwd_idx].to(pipeline.device)
            else:
                stage_input_buffer = pipeline._activation_buffer()
                stage_input = pipeline.comms.recv_forward_send_backward(
                    input_grad, stage_input_buffer
                )

            stage_output = pipeline.forward_microbatch(
                fwd_idx, model, stage_input, microbatch_y[fwd_idx]
            )
            fwd_idx += 1
            if fwd_idx == num_microbatches:
                pipeline.record_forward_completion()

    with record_function("1f1b/cooldown"):
        while bwd_idx < num_microbatches:
            if pipeline.dim.is_pp_last_stage:
                output_grad = None
            else:
                bwd_buffer = pipeline._activation_buffer()
                output_grad = pipeline.comms.recv_backward(bwd_idx, bwd_buffer)

            input_grad = pipeline.backward_microbatch(
                bwd_idx,
                output_grad,
                sync_gradients=bwd_idx == num_microbatches - 1,
            )
            if not pipeline.dim.is_pp_first_stage:
                pipeline.comms.send_backward(bwd_idx, input_grad)
            bwd_idx += 1
