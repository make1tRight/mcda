from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score)
from domain_adaptation import *
from sklearn.model_selection import LeavePOut
from sklearn.preprocessing import label_binarize
from collections import Counter
from logger import Logger
import os
from physiological_measures import *
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
# from reverse_validation import ReverseValidator
from torch.optim.lr_scheduler import CosineAnnealingLR


def empty_line():
    print("\n")

def run_training(data, cfg: Config, logger: Logger):
    """
    主训练逻辑（LOSO）
    """
    x = data.drop(columns=["subject_id", "MWL_Rating"]).values
    y = data["MWL_Rating"].values
    groups = data["subject_id"].values
    unique_subjects = np.unique(groups)

    lpo = LeavePOut(p=1)  # 留p个被试做测试集
    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    subject_metrics = {
        "subject_id": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "auc": []}

    for test_idx in lpo.split(unique_subjects):
        target_subject = unique_subjects[list(test_idx[1])]
        source_subject = unique_subjects[list(test_idx[0])]
        print(f"target_subject={target_subject}")

        train_mask = np.isin(groups, source_subject)
        test_mask = np.isin(groups, target_subject)

        x_train_origin, y_train_origin = x[train_mask], y[train_mask]
        print(f"x_train_origin.shape={x_train_origin.shape}, y_train_origin.shape={y_train_origin.shape}")
        x_test, y_test = x[test_mask], y[test_mask]
        # x_test_clean_mask = ~np.isnan(x_test).any(axis=1)
        # y_test_clean_mask = ~np.isnan(y_test)
        # test_clean_mask = x_test_clean_mask & y_test_clean_mask

        # x_test = x_test[test_clean_mask]
        # y_test = y_test[test_clean_mask]
        label_counter_test = Counter(y_test)
        if len(label_counter_test) < cfg.threshold_2b_test:
            logger.log("Skipping subject: {} due to insufficient classes in target domain: ", 
                       target_subject)
            logger.log("Found {}, expected {}.", 
                       len(label_counter_test), 
                       cfg.num_classes, 
                       level="WARN")
            continue

        # SMOTE平衡数据 - acc在类别不平衡的时候不一定能够反映实际性能
        logger.log(
            "target domain: {}: {}", 
            target_subject, 
            label_counter_test, 
            level="info")
        # (knn补缺失值)
        x_train_imputed = impute_missing_values_by_knn(x_train_origin, cfg.knn_k)
        x_test = impute_missing_values_by_knn(x_test, cfg.knn_k)
        # x_train, y_train = balance_data(x_train_imputed, y_train_origin, cfg.smote_seed)

        x_train = x_train_imputed
        y_train = y_train_origin
        logger.log(
            "input_dataset: source:{}, target:{}",
            x_train.shape,
            x_train.shape,
            level="info")

        # 模型初始化
        # 标准化
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        # 训练
        source_x = torch.tensor(x_train, dtype=torch.float32).to(cfg.device)
        source_y = torch.tensor(y_train, dtype=torch.long).to(cfg.device)
        target_x = torch.tensor(x_test, dtype=torch.float32).to(cfg.device)
        target_y = torch.tensor(y_test, dtype=torch.float32).to(cfg.device)
        assert not torch.isnan(source_x).any() and not torch.isinf(source_x).any()
        assert not torch.isnan(target_x).any() and not torch.isinf(target_x).any()

        # 模型初始化
        input_dim = source_x.shape[1]
        print(f"source_x.shape: {source_x.shape}")
        model_e = FeatureExtractor(input_dim).to(cfg.device)
        # model_e = TransformerEncoder(input_dim).to(cfg.device)
        # model_e = FeatureEncoderWithAttention(input_dim).to(cfg.device)
        model_c1 = Classifier(128, 64, 32, 3).to(cfg.device)
        model_c2 = Classifier(128, 256, 128, 3).to(cfg.device)
        model_d = DomainDiscriminator(128).to(cfg.device)

        optimizer_e = optim.Adam(model_e.parameters(), 
                                 lr=cfg.lr_encoder,
                                 weight_decay=1e-4)
        optimizer_c1 = optim.Adam(model_c1.parameters(), 
                                  lr=cfg.lr_classifier,
                                  weight_decay=1e-4)
        optimizer_c2 = optim.Adam(model_c2.parameters(), 
                                  lr=cfg.lr_classifier,
                                  weight_decay=1e-4)
        optimizer_d = optim.Adam(model_d.parameters(), 
                                 lr=cfg.lr_domain_discriminator,
                                 weight_decay=1e-4)
        source_dataset = TensorDataset(source_x, source_y)
        source_loader = DataLoader(source_dataset, batch_size=64, shuffle=True)
        target_dataset = TensorDataset(target_x, target_y)
        target_loader = DataLoader(target_dataset, batch_size=64, shuffle=True)

        loss_manager = LossManager()
        best_state = mcd_train(
            source_loader,
            target_loader,
            model_e,
            model_c1,
            model_c2,
            model_d,
            optimizer_e,
            optimizer_c1,
            optimizer_c2,
            optimizer_d,
            cfg,
            logger,
            loss_manager)
        model_e.load_state_dict(best_state["model_e"])
        model_c1.load_state_dict(best_state["model_c1"])
        model_c2.load_state_dict(best_state["model_c2"])
        model_e.eval()
        model_c1.eval()
        model_c2.eval()
        

        with torch.no_grad():
            features = model_e(target_x)
            pred = (model_c1(features) + model_c2(features)) / 2
            y_pred = pred.argmax(dim=1).cpu().numpy()
            y_prob = pred.cpu().numpy()

        # 积累所有被试的真实和预测标签
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)
        print(np.unique(y_test), y_test[:20])
        plot_confusion_matrix(
            y_test,
            y_pred,
            str(target_subject),
            os.path.join(cfg.log_path, f"{target_subject}"))
        # 指标计算
        # accuracy = balanced_accuracy_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, 
                                    average="weighted", 
                                    zero_division=0)
        recall = recall_score(y_test, y_pred, 
                              average="weighted", 
                              zero_division=0)
        f1 = f1_score(y_test, y_pred, 
                      average="weighted", 
                      zero_division=0)

        n_classes = len(np.unique(y_test))
        try:
            y_true_bin = label_binarize(y_test, 
                                        classes=list(range(n_classes)))
            y_prob_array = np.array(y_prob)
            auc = roc_auc_score(
                y_true_bin, y_prob_array, 
                average="weighted", 
                multi_class="ovr")
            logger.log("AUC: {:.2f}", 
                       auc * 100, 
                       level="INFO")
            subject_metrics["auc"].append(auc)
        except Exception as e:
            logger.log("AUC Calculation Error: {}", e, level="WARN")
        empty_line()
        logger.log("====== performance matrics {} ======", target_subject)
        logger.log("Accuracy: {:.2f}%", 
                   accuracy * 100, 
                   level="INFO")
        logger.log("Precision: {:.2f}%", 
                   precision * 100, 
                   level="INFO")
        logger.log("Recall: {:.2f}%", 
                   recall * 100, 
                   level="INFO")
        logger.log("F1 Score: {:.2f}%", 
                   f1 * 100, 
                   level="INFO")
        logger.log("====== performance matrics {} ======", target_subject)

        subject_metrics["subject_id"].append(target_subject)
        subject_metrics["accuracy"].append(accuracy)
        subject_metrics["precision"].append(precision)
        subject_metrics["recall"].append(recall)
        subject_metrics["f1"].append(f1)

        torch.cuda.empty_cache()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_confusion_matrix(
        all_y_true, 
        all_y_pred, 
        "All", 
        os.path.join(cfg.log_path, f"all_{timestamp}"))

    return subject_metrics

def main():
    try:
        cfg = Config()
        # 生理数据读取
        # full_df = load_eeg_data(cfg)
        logger = Logger(cfg.log_path)
        lbl_classifier = LabelClassifier(cfg.low_level, 
                                         cfg.mid_level, 
                                         cfg.high_level, 
                                         cfg.binary_threshold,
                                         cfg.num_classes)
        mm_loader = MultimodalLoader(cfg, logger, lbl_classifier)
        full_df = mm_loader.LoadMultimodalData()
        performance_list = []
        logger.log("info: {}", full_df.shape, level="INFO")

        performance = run_training(full_df, cfg, logger)
        performance_list.append(performance)
        metrics = ["accuracy", "precision", "recall", "f1", "auc"]
        logger.save_summary(performance_list, metrics)
    except KeyboardInterrupt:
        logger.log("Capture ctrl+c, exiting...")
    finally:
        logger.log("Exit successfully.")

if __name__ == "__main__":
    main()
