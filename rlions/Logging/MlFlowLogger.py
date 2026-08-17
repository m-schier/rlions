#  RLIonS - Reinforcement Learning Ion Shuttling Compiler
#  Copyright (C) 2026 Maximilian Schier, Lea Richtmann
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import os.path
import sys

import numpy as np

from .Logger import Logger, VideoObject, LogDictType, ImageObject

import mlflow


class MlFlowLogger(Logger):
    def __init__(self, uri='tmp/mlruns', config=None, experiment_id=None):
        super(MlFlowLogger, self).__init__()
        self.uri = uri
        self.run = None
        self.config = config
        self.experiment_id = experiment_id

    @property
    def local_storage(self):
        """
        Local artifact storage path
        """
        assert self.run
        # mlflow is quite broken when copying around the mlruns folder (for example from LUIS), we build this ourselves
        return os.path.join(self.uri, self.run.info.experiment_id, self.run.info.run_id, 'artifacts')

    def open(self):
        mlflow.set_tracking_uri(self.uri)
        self.run = mlflow.start_run(experiment_id=self.experiment_id)

        if self.config:
            mlflow.log_params(self.config)

        mlflow.set_tag("system.hostname", os.environ.get("HOSTNAME", "<UNKNOWN>"))
        mlflow.set_tag("slurm.job_id", os.environ.get("SLURM_JOB_ID", "<LOCAL>"))
        
        # Log command line
        path = os.path.join(self.local_storage, "cmdline.txt")
        print(f"Writing cmdline to {path}", file=sys.stderr)
        with open(path, 'w') as fp:
            print(*sys.argv, file=fp)

    def close(self):
        mlflow.end_run("FINISHED" if self._active_exception is None else "FAILED")
        self.run = None

    def log_dict(self, metrics: LogDictType, step: int):
        if self.run is None:
            raise ValueError("Must open() first")

        metrics_dict = {}

        for k, v in metrics.items():
            if isinstance(v, VideoObject):
                if self.uri.startswith('http:/'):
                    raise ValueError("Artifact logging for remote tracking not tested")

                from moviepy.editor import ImageSequenceClip

                # Probably some easier way to get this
                local_folder = os.path.join(self.local_storage, k)
                os.makedirs(local_folder, exist_ok=True)
                local_path = os.path.join(local_folder, f"{step}.mp4")
                clip = ImageSequenceClip(list(v.frames), fps=v.fps)
                clip.write_videofile(local_path, fps=v.fps, codec='libx264')
            elif isinstance(v, ImageObject):
                mlflow.log_image(v.rgb_array, key=k, step=step)
            elif v is None:
                # MlFlow does not support literal `None` and crashes if we put np.nan, so just drop it
                pass
            else:
                metrics_dict[k] = float(v)

        mlflow.log_metrics(metrics_dict, step=step)
