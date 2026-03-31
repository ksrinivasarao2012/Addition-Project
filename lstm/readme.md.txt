cd C:\Users\heman\Music\rl_imu_project
carla_env37\Scripts\activate
pip install torch numpy pandas matplotlib scipy
python lstm\train_lstm.py


Here is exactly what the script does and why each decision was made:
Data → Windows. Each window is 50 timesteps (2.5 s at 20 Hz). The target is gt_pos[t+50] − gt_pos[t] — the true 2D displacement over that window. CARLA ground truth is always available everywhere including inside the tunnel, so training is fully supervised with no gaps.
Split. Chronological — oldest 75% for training, next 15% validation, most recent 10% test. Never shuffle across boundaries because shuffling would leak future IMU patterns into training, making validation accuracy falsely optimistic.
Normalisation. Mean and std computed only from the training set, then applied to val and test. Saved to models/lstm_norm_params.npz — the KF-LSTM fusion module will load this same file at inference time to normalise incoming IMU windows identically.
Model. 2-layer LSTM (hidden=128) followed by Linear(128→64)→ReLU→Dropout→Linear(64→32)→ReLU→Linear(32→2). Dropout only between LSTM layers and in the MLP head, not after the final output layer. Gradient clipping at norm=1.0 to prevent the instability that LSTM training occasionally shows.
Training. Adam with ReduceLROnPlateau — halves the LR when val loss stops improving for 8 epochs. Early stops after 20 epochs without improvement. Best checkpoint is saved immediately whenever validation loss improves, so Ctrl-C at any point still leaves you with the best weights found so far.
Outputs. lstm_drift.pt (weights + hyperparameters), lstm_norm_params.npz (normalisation), lstm_training_log.csv (every epoch's loss/rmse/lr), and two diagnostic plots.