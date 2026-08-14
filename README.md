In train process, it uses MSE loss, while in public and private test, it uses RMSE loss to compute and record. In this way, it guarantees train consistency since it uses diff component in computing loss. At the same time, it enhances interpretability in public and private test by using RMSE loss.

This dataset involves train-20240123-14902.csv, publictest-20240508.csv and privatetest-20240506-yh.csv, which could be used for public and private test respectively. 14902, 1298 and 3589 samples are for train, public and private dataset at present.
