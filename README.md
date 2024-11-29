# mt-logs

## Dependencies

- numpy, scipy, pandas, matplotlib, seaborn

## example

```bash
python3 uc_plotter.py
```

## Configuration

```python
uc1_run_dir = "freddy_uc1_log"
uc2_run_dir = "freddy_uc2_align_log"
```

```python
uc1_plotter = UCPlotter(uc1_run_dir)
uc1_plotter.plot_uc1_ts(use_post_proc=True)
```

- `use_post_proc` : If True, the post-processed data will be used for plotting. Otherwise, the raw data will be used.
- The post-processed data steps:

  - Clipping for UC1 platform force.

    ```python
    new_pf_cmd = clip(pf_cmd, -10, 10)
    ```

  - PI controller on platform velocty.

    ```python
    pf_cmd = left_2dw_at_base + right_2dw_at_base
    PI_cmd = PI(pf_vel_sp, pf_vel_current)
    new_pf_cmd = pf_cmd + PI_cmd
    ```

> [!IMPORTANT]
> Remarks: Now that I think about it, The below runs are wrong as the PI
  controller is not implemented correctly on the platform velocity. The PI
  controller will always try to maintain the velocity at the setpoint. :(
