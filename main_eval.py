import polars as pl
import numpy as np
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt


class LongitudinalEvaluator:
    def __init__(self, real_df: pl.DataFrame, synth_df: pl.DataFrame,
                 patient_id: str, time_col: str):
        self.real = real_df
        self.synth = synth_df
        self.pid = patient_id
        self.tcol = time_col

    # ---------- 1. Marginali ----------
    def marginal_stats(self, columns):
        results = {}
        for col in columns:
            r = self.real[col].drop_nulls().to_numpy()
            s = self.synth[col].drop_nulls().to_numpy()
            ks = ks_2samp(r, s).statistic
            results[col] = {
                "real_mean": r.mean(),
                "synth_mean": s.mean(),
                "real_std": r.std(),
                "synth_std": s.std(),
                "ks": ks
            }
        return results

    # ---------- 2. Correlazioni ----------
    def correlation_matrix(self, columns):
        real_corr = self.real.select(columns).to_pandas().corr()
        synth_corr = self.synth.select(columns).to_pandas().corr()
        return real_corr, synth_corr

    # ---------- 3. Traiettorie ----------
    def plot_mean_trajectories(self, biomarkers):
        for bm in biomarkers:
            real_mean = (
                self.real
                .groupby(self.tcol)
                .agg(pl.mean(bm))
                .sort(self.tcol)
            )
            synth_mean = (
                self.synth
                .groupby(self.tcol)
                .agg(pl.mean(bm))
                .sort(self.tcol)
            )

            plt.figure()
            plt.plot(real_mean[self.tcol], real_mean[bm], label="Real")
            plt.plot(synth_mean[self.tcol], synth_mean[bm], label="Synthetic")
            plt.title(f"Mean trajectory: {bm}")
            plt.legend()
            plt.show()

    # ---------- 4. Trajectory similarity (DTW-like proxy) ----------
    def trajectory_variance(self, biomarker):
        def traj_var(df):
            grouped = df.groupby(self.pid)
            slopes = []
            for _, g in grouped:
                g = g.sort(self.tcol)
                if g.height > 1:
                    slope = np.polyfit(
                        g[self.tcol].to_numpy(),
                        g[biomarker].to_numpy(), 1
                    )[0]
                    slopes.append(slope)
            return np.array(slopes)

        return {
            "real": traj_var(self.real),
            "synthetic": traj_var(self.synth)
        }

    # ---------- 5. Event sanity check ----------
    def event_rates(self, event_col):
        def rate(df):
            return (
                df.groupby(self.pid)
                .agg(pl.max(event_col))
                .select(pl.mean(event_col))
                .item()
            )

        return {
            "real_event_rate": rate(self.real),
            "synth_event_rate": rate(self.synth)
        }
