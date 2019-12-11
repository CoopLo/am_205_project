# Missile simulation and interception

1. Visualise enemy missile tracking
```
python3 tracking.py
```
Lagrange interpolation and Kalman filtering for tracking enemy missile trajectories based on incomplete sensor measurements. All parameters of the trajectories can be set within the file.

2. Compare tracking methods
```
python3 tracking_comp.py num_trials
```
Run Lagrange interpolation and Kalman filtering `num_trials` times for different enemy missile trajectories, and obtain MSEs for both methods.

3. Optimisation-based interception
```
python3 optimisation_intercept.py
```
Run optimisation-based framework for enemy missile interception with response missile. All parameters of the trajectories can be set within the file.

4. Numerical-based interception
```
python3 enemy_response.py
```
Run numerical-based framework for enemy missile interception. Runs `enemy.py` and `response.py` in parallel, which handle enemy missile trajectory projection and numerical-based response missile respectively. Generates plots of interception. All parameters of the trajectories can be set within the file.
