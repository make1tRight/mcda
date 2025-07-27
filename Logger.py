import os
import logging
import sys
import numpy as np
from datetime import datetime


class Logger:
    def __init__(self, log_dir, name=__name__, log_name_prefix="log", level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self._get_log_level(level))

        self.logger.propagate = False
        if self.logger.handlers:
            self.logger.handlers.clear()

        # 日志路径设置
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_path = os.path.join(log_dir, f"{log_name_prefix}_{timestamp}.log")

        # 文件输出
        file_formatter = logging.Formatter(
            "[{asctime}][{levelname}] {message}",
            datefmt="%Y-%m-%d %H:%M:%S",
            style='{')
        # file_formatter = logging.Formatter(
        #     "[{levelname}] {message}",
        #     datefmt="%Y-%m-%d %H:%M:%S",
        #     style='{')

        file_handler = logging.FileHandler(self.log_path, encoding='utf-8')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # 控制台输出
        console_formatter = logging.Formatter("{message}", style='{')
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

    def _get_log_level(self, level: str = "INFO") -> int:
        if isinstance(level, int):
            return level
        if isinstance(level, str):
            level = level.upper()
            return {
                "DEBUG": logging.DEBUG,
                "INFO": logging.INFO,
                "WARNING": logging.WARNING,
                "ERROR": logging.ERROR,
                "CRITICAL": logging.CRITICAL
                }.get(level, logging.INFO)

    def log(self, message: str, *args, level: str = "INFO", **kwargs):
        log_level_int = self._get_log_level(level)
        if log_level_int >= self.logger.level:
            try:
                formatted_message = message.format(*args, **kwargs)
            except Exception as e:
                formatted_message = f"[FormatError] {message} | Args: {args} | Kwargs: {kwargs} | Error: {e}"

        self.logger.log(log_level_int, 
                        formatted_message)

    def log_metrics(self, subject_id, metrics: dict, level="info"):
        msg = f"Subject {subject_id} |" + " | ".join(
            f"{k}: {v * 100: .2f}" for k, v in metrics.items()
        )
        self.log(msg, level=level)

    def save_summary(self, performance_list, metrics):
        self.log("\n=== Summary of Repetitions ===")
        for metric in metrics:
            values = [perf[metric] for perf in performance_list]
            mean = np.mean(values)
            std = np.std(values)
            self.log(f"{metric.capitalize():<9}: {mean * 100:.2f}% ± {std * 100:.2f}%")

    def get_log_path(self):
        return self.log_path
