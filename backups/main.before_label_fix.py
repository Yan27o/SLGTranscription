from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D

# 论文中描述的标签范围：0~23（A-Y，排除 J、Z）
VALID_LABELS = set(range(24))
ALPHABET_NO_JZ = [ch for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if ch not in {"J", "Z"}]
ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT_DIR / "mnt_dataset"
DEFAULT_MODEL_PATH = ROOT_DIR / "sign_language_cnn.keras"
DEFAULT_FIGURE_PATH = ROOT_DIR / "training_history.png"
DEFAULT_HAND_MODEL_PATH = ROOT_DIR / "models" / "hand_landmarker.task"


def label_to_char(label: int) -> str:
    if 0 <= label < len(ALPHABET_NO_JZ):
        return ALPHABET_NO_JZ[label]
    return f"cls_{label}"

# 模块 2：数据集处理
def load_and_process_data(train_csv: Path, test_csv: Path):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    train_df = train_df[train_df["label"].isin(VALID_LABELS)].copy()
    test_df = test_df[test_df["label"].isin(VALID_LABELS)].copy()

    y_train = train_df["label"].to_numpy(dtype=np.int32)
    y_test = test_df["label"].to_numpy(dtype=np.int32)

    x_train = train_df.drop(columns=["label"]).to_numpy(dtype=np.float32) / 255.0
    x_test = test_df.drop(columns=["label"]).to_numpy(dtype=np.float32) / 255.0

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    print("=== 数据集信息 ===")
    print(f"训练集: {x_train.shape}, 标签范围: {y_train.min()} ~ {y_train.max()}")
    print(f"测试集: {x_test.shape}, 标签范围: {y_test.min()} ~ {y_test.max()}")
    print(f"训练标签类别数: {len(np.unique(y_train))}")

    return x_train, y_train, x_test, y_test

# 模块 3：CNN 模型构建
def build_model() -> tf.keras.Model:
    model = Sequential(
        [
            # 卷积层1：32个3×3卷积核，ReLU激活（提取基础特征：边缘、纹理）
            Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
            # 池化层1：2×2池化（降维，减少计算量）
            MaxPooling2D((2, 2)),
            # 卷积层2：64个3×3卷积核（提取更复杂的特征：手部轮廓）
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            # 卷积层3：128个3×3卷积核（提取精细特征：手指姿态）
            Conv2D(128, (3, 3), activation="relu"),
            # 展平层：把2D特征图转成1D向量（连接卷积层和全连接层）
            Flatten(),
            # 全连接层：256个神经元（整合特征）
            Dense(256, activation="relu"),
            # Dropout：随机丢弃50%神经元（防止过拟合）
            Dropout(0.5),
            # 输出层：25个神经元（对应25个字母），Softmax激活（输出概率）
            Dense(24, activation="softmax"),
        ]
    )

    # 编译模型：指定优化器、损失函数、评估指标
    model.compile(
        optimizer="adam",   # Adam优化器（自适应学习率，训练更快）
        loss="sparse_categorical_crossentropy", # 稀疏交叉熵（标签是整数时用）
        metrics=["accuracy"],   # 监控准确率
    )
    return model

# 模块 4：模型训练
def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    batch_size: int,
):
    model = build_model()
    # 早停策略：验证损失5轮不下降则停止（防止过拟合）
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1,
    )
    return model, history

# 模块 5：模型评估
def evaluate_model(
    model: tf.keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
):
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    y_prob = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print("\n=== 测试集评估 ===")
    print(f"Test Loss: {loss:.5f}")
    print(f"Test Accuracy: {accuracy:.5f}")
    print(f"Macro Precision: {precision:.5f}")
    print(f"Macro Recall: {recall:.5f}")
    print(f"Macro F1: {f1:.5f}")

    return {"loss": loss, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# 可视化训练曲线（准确率+损失）
def save_training_figure(history: tf.keras.callbacks.History, figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    plt.close()
    print(f"\n训练曲线已保存: {figure_path}")

# 模块 6：实时推理
class RealTimeClassifier:
    def __init__(
        self,
        model_path: Path,
        confidence_threshold: float = 0.8,
        speak_cooldown_sec: float = 1.5,
        enable_speech: bool = False,
    ):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("手部跟踪需要 mediapipe，请在当前 Python 环境安装该依赖。") from exc

        if not DEFAULT_HAND_MODEL_PATH.is_file():
            raise FileNotFoundError(f"缺少手部跟踪模型: {DEFAULT_HAND_MODEL_PATH}")
        self.mp = mp
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(DEFAULT_HAND_MODEL_PATH),
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hands = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.last_frame_timestamp_ms = -1

        self.confidence_threshold = confidence_threshold

        self.tts = None
        if enable_speech:
            try:
                import pyttsx3

                self.tts = pyttsx3.init()
            except Exception as exc:
                print(f"警告：语音初始化失败，继续使用画面识别: {exc}")
        self.last_spoken: str | None = None
        self.last_spoken_time = 0.0
        self.speak_cooldown_sec = speak_cooldown_sec

    @staticmethod
    def preprocess_roi(roi_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        return normalized.reshape(1, 28, 28, 1)

    def predict(self, roi_bgr: np.ndarray):
        input_tensor = self.preprocess_roi(roi_bgr)
        probs = np.asarray(self.model(input_tensor, training=False))[0]
        pred_label = int(np.argmax(probs))
        confidence = float(np.max(probs))
        return pred_label, label_to_char(pred_label), confidence

    def maybe_speak(self, char_label: str, confidence: float) -> None:
        now = time.time()
        if self.tts is None or confidence < self.confidence_threshold:
            return
        if self.last_spoken == char_label and (now - self.last_spoken_time) < self.speak_cooldown_sec:
            return

        self.tts.say(char_label)
        self.tts.runAndWait()
        self.last_spoken = char_label
        self.last_spoken_time = now

    def detect_hand(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        # VIDEO 模式要求时间戳严格递增，使用单调时钟避免系统时间调整影响。
        timestamp_ms = max(time.monotonic_ns() // 1_000_000,
                           self.last_frame_timestamp_ms + 1)
        self.last_frame_timestamp_ms = timestamp_ms
        result = self.hands.detect_for_video(image, timestamp_ms)
        return result.hand_landmarks[0] if result.hand_landmarks else None

    @staticmethod
    def hand_bounds(landmarks, width: int, height: int):
        xs = [lm.x * width for lm in landmarks]
        ys = [lm.y * height for lm in landmarks]
        # 按手部大小留出边距，裁剪仅供模型使用，不在画面上显示方框。
        padding = max(20, int(max(max(xs) - min(xs), max(ys) - min(ys)) * 0.15))
        return (max(0, int(min(xs)) - padding),
                max(0, int(min(ys)) - padding),
                min(width, int(max(xs)) + padding),
                min(height, int(max(ys)) + padding))

    def annotate_frame(self, frame: np.ndarray) -> np.ndarray:
        landmarks = self.detect_hand(frame)
        if landmarks is None:
            cv2.putText(frame, "No hand detected", (15, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            return frame

        h, w = frame.shape[:2]
        x_min, y_min, x_max, y_max = self.hand_bounds(landmarks, w, h)
        if x_max <= x_min or y_max <= y_min:
            return frame
        # 先识别原始图像，再绘制骨架，避免线条污染模型输入。
        _, pred_char, conf = self.predict(frame[y_min:y_max, x_min:x_max])
        points = [(int(np.clip(lm.x * w, 0, w - 1)),
                   int(np.clip(lm.y * h, 0, h - 1))) for lm in landmarks]
        connections = self.mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
        for connection in connections:
            cv2.line(frame, points[connection.start], points[connection.end],
                     (80, 230, 80), 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(frame, point, 4, (0, 180, 255), -1, cv2.LINE_AA)

        label = f"{pred_char if conf >= self.confidence_threshold else '?'} ({conf:.1%})"
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        anchor = (max(0, min(x_min, w - text_width - 5)),
                  min(h - 5, max(y_min - 10, text_height + 40)))
        cv2.putText(frame, label, anchor, cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2, cv2.LINE_AA)
        self.maybe_speak(pred_char, conf)
        return frame

    def classify_from_camera(self, camera_index: int = 0) -> None:
        cap = cv2.VideoCapture(camera_index)
        window_name = "Real-Time Sign Language Classification"
        try:
            if not cap.isOpened():
                raise RuntimeError(
                    f"无法打开摄像头 {camera_index}，请检查系统相机权限、关闭占用摄像头的软件，"
                    "或使用 --camera-index 1 选择其他摄像头。"
                )
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 960, 720)
            print("手掌跟踪已启动，手可在画面内自由移动；按 Q、Esc 或关闭窗口退出。")
            while True:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("读取摄像头画面失败，请检查摄像头连接后重新运行。")

                frame = cv2.flip(frame, 1)
                frame = self.annotate_frame(frame)
                cv2.putText(frame, "Q / Esc: Quit | Hand tracking", (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            if self.hands is not None:
                self.hands.close()
            if self.tts is not None:
                self.tts.stop()

# 模块1：命令行参数解析：允许运行时自定义参数（无需改代码）
def parse_args():
    parser = argparse.ArgumentParser(description="Sign Language Gestures to Speech Reproduction")
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_DATA_DIR / "sign_mnist_train.csv")
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_DATA_DIR / "sign_mnist_test.csv")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_FIGURE_PATH)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument("--camera-index", type=int, default=0, help="摄像头编号，默认 0")
    parser.add_argument("--speak", action="store_true", help="开启语音播报（可能降低画面流畅度）")
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "realtime"],
        default="realtime",
        help="train: 训练并评估；eval: 仅评估现有模型；realtime: 摄像头实时识别",
    )
    args = parser.parse_args()
    if not 0 <= args.confidence_threshold <= 1:
        parser.error("--confidence-threshold 必须在 0 到 1 之间")
    if args.camera_index < 0:
        parser.error("--camera-index 不能为负数")
    return args

# 模块 7：主函数（代码入口）
def main():
    args = parse_args()

    if args.mode == "train":
        x_train, y_train, x_test, y_test = load_and_process_data(args.train_csv, args.test_csv)
        model, history = train_model(
            x_train=x_train,
            y_train=y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        evaluate_model(model, x_test, y_test)
        args.model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(args.model_path)
        print(f"模型已保存: {args.model_path}")
        try:
            save_training_figure(history, args.figure_path)
        except Exception as exc:
            print(f"警告：训练曲线保存失败（不影响模型使用）: {exc}")

    elif args.mode == "eval":
        if not args.model_path.exists():
            raise FileNotFoundError(f"模型不存在: {args.model_path}，请先运行 train 模式。")
        _, _, x_test, y_test = load_and_process_data(args.train_csv, args.test_csv)
        model = tf.keras.models.load_model(args.model_path)
        evaluate_model(model, x_test, y_test)

    elif args.mode == "realtime":
        if not args.model_path.exists():
            raise FileNotFoundError(f"模型不存在: {args.model_path}，请先运行 train 模式。")
        rtc = RealTimeClassifier(
            model_path=args.model_path,
            confidence_threshold=args.confidence_threshold,
            enable_speech=args.speak,
        )
        rtc.classify_from_camera(camera_index=args.camera_index)


if __name__ == "__main__":
    main()
