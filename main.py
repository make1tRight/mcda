# import matplotlib.pyplot as plt
# from pydantic import Discriminator
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             roc_auc_score)
from DomainAdaptation import *
# import PhysiologicalMeasures
from sklearn.model_selection import LeavePOut
from sklearn.preprocessing import label_binarize
from collections import Counter
from Logger import Logger
import os
from PhysiologicalMeasures import *


def empty_line():
    print("\n")

# ------------------------------
# 主训练逻辑（LOSO）
# ------------------------------
# def run_lotso_training(data):
#     x = data.drop(columns=["subject_id", "MWL_Rating"]).values
#     y = data["MWL_Rating"].values
#     groups = data["subject_id"].values
#     unique_subjects = np.unique(groups)
#
#     lpo = LeavePOut(p=2)  # 留两个被试做测试集
#     all_y_true = []
#     all_y_pred = []
#     all_y_prob = []
#     subject_metrics = {
#         "subject_id": [],
#         "accuracy": [],
#         "precision": [],
#         "recall": [],
#         "f1": [],
#         "auc": [],
#     }
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     for test_idx in lpo.split(unique_subjects):
#         target_subject = unique_subjects[list(test_idx[1])]
#         source_subject = unique_subjects[list(test_idx[0])]
#
#         train_mask = np.isin(groups, source_subject)
#         test_mask = np.isin(groups, target_subject)
#
#         x_train_origin, y_train_origin = x[train_mask], y[train_mask]
#         x_test, y_test = x[test_mask], y[test_mask]
#         x_test_clean_mask = ~np.isnan(x_test).any(axis=1)
#         y_test_clean_mask = ~np.isnan(y_test)
#         test_clean_mask = x_test_clean_mask & y_test_clean_mask
#
#         x_test = x_test[test_clean_mask]
#         y_test = y_test[test_clean_mask]
#         label_counter_test = Counter(y_test)
#
#         # SMOTE平衡数据 - acc在类别不平衡的时候不一定能够反映实际性能
#         print(f"target domain: {target_subject}: {label_counter_test}")
#         # (knn补缺失值)
#         x_train_imputed = PhysiologicalMeasures.impute_missing_values_by_knn(x_train_origin)
#         x_train, y_train = PhysiologicalMeasures.balance_data(x_train_imputed, y_train_origin)
#         print(f"input_dataset: source{x_train.shape}, target{x_test.shape}")
#
#         # 模型初始化
#         input_dim = x.shape[1]
#         # model_e = FeatureExtractor(input_dim).to(device)
#         model_e = TransformerFeatureExtractor(input_dim).to(device)
#         model_c1 = Classifier(128, 64, 32, 3).to(device)
#         model_c2 = Classifier(128, 256, 128, 3).to(device)
#         model_d = DomainDiscriminator(128).to(device)
#
#         optimizer_e = optim.Adam(model_e.parameters(), lr=1e-3)
#         optimizer_c1 = optim.Adam(model_c1.parameters(), lr=1e-3)
#         optimizer_c2 = optim.Adam(model_c2.parameters(), lr=1e-3)
#         optimizer_d = optim.Adam(model_d.parameters(), lr=1e-4)
#
#         # 标准化
#         scaler = StandardScaler()
#         x_train = scaler.fit_transform(x_train)
#         x_test = scaler.transform(x_test)
#
#         # 训练
#         source_x = torch.tensor(x_train, dtype=torch.float32).to(device)
#         source_y = torch.tensor(y_train, dtype=torch.long).to(device)
#         target_x = torch.tensor(x_test, dtype=torch.float32).to(device)
#         assert not torch.isnan(source_x).any() and not torch.isinf(source_x).any()
#         assert not torch.isnan(target_x).any() and not torch.isinf(target_x).any()
#
#         if torch.isnan(source_x).any() or torch.isinf(source_x).any():
#             print("source_x contains NaN or Inf!")
#         loss_manager = LossManager.LossManager()
#         for epoch in range(1):
#             mcd_train_2(source_x, source_y, target_x,
#                         model_e, model_c1, model_c2, model_d, optimizer_d,
#                         optimizer_e, optimizer_c1, optimizer_c2,
#                         500, 5, 150, 100,
#                         loss_manager)
#         # 测试
#         model_e.eval()
#         model_c1.eval()
#         model_c2.eval()
#
#         with torch.no_grad():
#             features = model_e(target_x)
#             pred = (model_c1(features) + model_c2(features)) / 2
#             y_pred = pred.argmax(dim=1).cpu().numpy()
#             y_prob = pred.cpu().numpy()
#
#         # 积累所有被试的真实和预测标签
#         all_y_true.extend(y_test)
#         all_y_pred.extend(y_pred)
#         all_y_prob.extend(y_prob)
#         PhysiologicalMeasures.plot_confusion_matrix(y_test, y_pred, str(target_subject))
#         # 指标计算
#         accuracy = accuracy_score(y_test, y_pred)
#         precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
#         recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
#         f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
#
#         n_classes = len(np.unique(y_test))
#         try:
#             y_true_bin = label_binarize(y_test, classes=list(range(n_classes)))
#             y_prob_array = np.array(y_prob)
#             auc = roc_auc_score(y_true_bin, y_prob_array, average='macro', multi_class='ovr')
#             print(f"\nAUC: {auc * 100:.2f}%")
#             subject_metrics["auc"].append(auc)
#         except Exception as e:
#             print(f"\nAUC Calculation Error: {e}")
#         empty_line()
#         print(f"Accuracy: {accuracy * 100:.2f}%")
#         print(f"Precision: {precision * 100:.2f}%")
#         print(f"Recall: {recall * 100:.2f}%")
#         print(f"F1 Score: {f1 * 100:.2f}%")
#         empty_line()
#
#         subject_metrics["subject_id"].append(target_subject)
#         subject_metrics["accuracy"].append(accuracy)
#         subject_metrics["precision"].append(precision)
#         subject_metrics["recall"].append(recall)
#         subject_metrics["f1"].append(f1)
#
#         torch.cuda.empty_cache()
#     PhysiologicalMeasures.plot_confusion_matrix(all_y_true, all_y_pred, "All")
#
#     return subject_metrics


# def run_lotso_training(data):
#     x = data.drop(columns=["subject_id", "MWL_Rating"]).values
#     y = data["MWL_Rating"].values
#     groups = data["subject_id"].values
#     unique_subjects = np.unique(groups)
#
#     lpo = LeavePOut(p=2)  # 留p个被试做测试集
#     all_y_true = []
#     all_y_pred = []
#     all_y_prob = []
#     subject_metrics = {
#         "subject_id": [],
#         "accuracy": [],
#         "precision": [],
#         "recall": [],
#         "f1": [],
#         "auc": [],
#     }
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     for test_idx in lpo.split(unique_subjects):
#         target_subject = unique_subjects[list(test_idx[1])]
#         source_subject = unique_subjects[list(test_idx[0])]
#
#         train_mask = np.isin(groups, source_subject)
#         test_mask = np.isin(groups, target_subject)
#
#         x_train_origin, y_train_origin = x[train_mask], y[train_mask]
#         x_test, y_test = x[test_mask], y[test_mask]
#         x_test_clean_mask = ~np.isnan(x_test).any(axis=1)
#         y_test_clean_mask = ~np.isnan(y_test)
#         test_clean_mask = x_test_clean_mask & y_test_clean_mask
#
#         x_test = x_test[test_clean_mask]
#         y_test = y_test[test_clean_mask]
#         label_counter_test = Counter(y_test)
#
#         # SMOTE平衡数据 - acc在类别不平衡的时候不一定能够反映实际性能
#         print(f"target domain: {target_subject}: {label_counter_test}")
#         # (knn补缺失值)
#         x_train_imputed = PhysiologicalMeasures.impute_missing_values_by_knn(x_train_origin)
#         x_train, y_train = PhysiologicalMeasures.balance_data(x_train_imputed, y_train_origin)
#         print(f"input_dataset: source{x_train.shape}, target{x_test.shape}")
#
#         # 模型初始化
#         input_dim = x.shape[1]
#         # model_e = FeatureExtractor(input_dim).to(device)
#         model_e = TransformerFeatureExtractor(input_dim).to(device)
#         model_c1 = Classifier(128, 64, 32, 3).to(device)
#         model_c2 = Classifier(128, 256, 128, 3).to(device)
#         model_d = DomainDiscriminator(128).to(device)
#
#         optimizer_e = optim.Adam(model_e.parameters(), lr=1e-3)
#         optimizer_c1 = optim.Adam(model_c1.parameters(), lr=1e-3)
#         optimizer_c2 = optim.Adam(model_c2.parameters(), lr=1e-3)
#         optimizer_d = optim.Adam(model_d.parameters(), lr=1e-4)
#
#         # 标准化
#         scaler = StandardScaler()
#         x_train = scaler.fit_transform(x_train)
#         x_test = scaler.transform(x_test)
#
#         # 训练
#         source_x = torch.tensor(x_train, dtype=torch.float32).to(device)
#         source_y = torch.tensor(y_train, dtype=torch.long).to(device)
#         target_x = torch.tensor(x_test, dtype=torch.float32).to(device)
#         assert not torch.isnan(source_x).any() and not torch.isinf(source_x).any()
#         assert not torch.isnan(target_x).any() and not torch.isinf(target_x).any()
#
#         if torch.isnan(source_x).any() or torch.isinf(source_x).any():
#             print("source_x contains NaN or Inf!")
#         loss_manager = LossManager.LossManager()
#         for epoch in range(1):
#             mcd_train_2(source_x, source_y, target_x,
#                         model_e, model_c1, model_c2, model_d, optimizer_d,
#                         optimizer_e, optimizer_c1, optimizer_c2,
#                         300, 5, 150, 100,
#                         loss_manager)
#         # FeaturesExtractor可以设置step1_iter=500
#         # TransformerExtractor设置step1_iter=300就差不多了
#         # 测试
#         model_e.eval()
#         model_c1.eval()
#         model_c2.eval()
#
#         with torch.no_grad():
#             features = model_e(target_x)
#             pred = (model_c1(features) + model_c2(features)) / 2
#             y_pred = pred.argmax(dim=1).cpu().numpy()
#             y_prob = pred.cpu().numpy()
#
#         # 积累所有被试的真实和预测标签
#         all_y_true.extend(y_test)
#         all_y_pred.extend(y_pred)
#         all_y_prob.extend(y_prob)
#         PhysiologicalMeasures.plot_confusion_matrix(y_test, y_pred, str(target_subject))
#         # 指标计算
#         accuracy = accuracy_score(y_test, y_pred)
#         precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
#         recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
#         f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
#
#         n_classes = len(np.unique(y_test))
#         try:
#             y_true_bin = label_binarize(y_test, classes=list(range(n_classes)))
#             y_prob_array = np.array(y_prob)
#             auc = roc_auc_score(y_true_bin, y_prob_array, average='macro', multi_class='ovr')
#             print(f"\nAUC: {auc * 100:.2f}%")
#             subject_metrics["auc"].append(auc)
#         except Exception as e:
#             print(f"\nAUC Calculation Error: {e}")
#         empty_line()
#         print(f"Accuracy: {accuracy * 100:.2f}%")
#         print(f"Precision: {precision * 100:.2f}%")
#         print(f"Recall: {recall * 100:.2f}%")
#         print(f"F1 Score: {f1 * 100:.2f}%")
#         empty_line()
#
#         subject_metrics["subject_id"].append(target_subject)
#         subject_metrics["accuracy"].append(accuracy)
#         subject_metrics["precision"].append(precision)
#         subject_metrics["recall"].append(recall)
#         subject_metrics["f1"].append(f1)
#
#         torch.cuda.empty_cache()
#     PhysiologicalMeasures.plot_confusion_matrix(all_y_true, all_y_pred, "All")
#
#     return subject_metrics

def run_lotso_training(data):
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
        "auc": [],
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for test_idx in lpo.split(unique_subjects):
        target_subject = unique_subjects[list(test_idx[1])]
        source_subject = unique_subjects[list(test_idx[0])]

        train_mask = np.isin(groups, source_subject)
        test_mask = np.isin(groups, target_subject)

        x_train_origin, y_train_origin = x[train_mask], y[train_mask]
        x_test, y_test = x[test_mask], y[test_mask]
        x_test_clean_mask = ~np.isnan(x_test).any(axis=1)
        y_test_clean_mask = ~np.isnan(y_test)
        test_clean_mask = x_test_clean_mask & y_test_clean_mask

        x_test = x_test[test_clean_mask]
        y_test = y_test[test_clean_mask]
        label_counter_test = Counter(y_test)

        # SMOTE平衡数据 - acc在类别不平衡的时候不一定能够反映实际性能
        # print(f"target domain: {target_subject}: {label_counter_test}")
        logger.log("target domain: {}: {}", target_subject, label_counter_test, level="info")
        # (knn补缺失值)
        # x_train_imputed = PhysiologicalMeasures.impute_missing_values_by_knn(x_train_origin)
        # x_train, y_train = PhysiologicalMeasures.balance_data(x_train_imputed, y_train_origin)
        x_train_imputed = impute_missing_values_by_knn(x_train_origin)
        x_train, y_train = balance_data(x_train_imputed, y_train_origin)
        # print(f"input_dataset: source{x_train.shape}, target{x_test.shape}")
        logger.log("input_dataset: source:{}, target:{}", x_train.shape, x_train.shape, level="info")

        # 模型初始化
        input_dim = x.shape[1]
        print(f"x.shape: {x.shape}")
        # model_e = FeatureExtractor(input_dim).to(device)
        model_e = FeatureEncoder(input_dim).to(device)
        # model_e = TransformerFeatureExtractor(input_dim).to(device)
        model_c1 = Classifier(128, 64, 32, 3).to(device)
        model_c2 = Classifier(128, 256, 128, 3).to(device)
        model_d = DomainDiscriminator(128).to(device)

        optimizer_e = optim.Adam(model_e.parameters(), lr=1e-3)
        optimizer_c1 = optim.Adam(model_c1.parameters(), lr=1e-3)
        optimizer_c2 = optim.Adam(model_c2.parameters(), lr=1e-3)
        optimizer_d = optim.Adam(model_d.parameters(), lr=1e-4)

        # 标准化
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        # 训练
        source_x = torch.tensor(x_train, dtype=torch.float32).to(device)
        source_x = source_x.unsqueeze(0)
        source_x = source_x.permute(0, 2, 1)
        source_y = torch.tensor(y_train, dtype=torch.long).to(device)
        target_x = torch.tensor(x_test, dtype=torch.float32).to(device)
        assert not torch.isnan(source_x).any() and not torch.isinf(source_x).any()
        assert not torch.isnan(target_x).any() and not torch.isinf(target_x).any()

        if torch.isnan(source_x).any() or torch.isinf(source_x).any():
            # print("source_x contains NaN or Inf!")
            logger.log("source_x contains NaN or Inf!", level="error")
        loss_manager = LossManager.LossManager()
        for epoch in range(1):
            mcd_train_3(source_x, source_y, target_x,
                        model_e, model_c1, model_c2, model_d, optimizer_d,
                        optimizer_e, optimizer_c1, optimizer_c2,
                        300, 5, 150, 100,
                        logger, loss_manager)
        # FeaturesExtractor可以设置step1_iter=500
        # TransformerExtractor设置step1_iter=300就差不多了
        # 测试
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
        plot_confusion_matrix(y_test, y_pred,
                              str(target_subject),
                              os.path.join(log_dir, f"{target_subject}"))
        # 指标计算
        accuracy = accuracy_score(y_test, 
                                  y_pred)
        precision = precision_score(y_test, 
                                    y_pred, 
                                    average='weighted', 
                                    zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        n_classes = len(np.unique(y_test))
        try:
            y_true_bin = label_binarize(y_test, classes=list(range(n_classes)))
            y_prob_array = np.array(y_prob)
            auc = roc_auc_score(y_true_bin, y_prob_array, average='macro', multi_class='ovr')
            # print(f"\nAUC: {auc * 100:.2f}%")
            logger.log("AUC: {:.2f}", auc * 100, level="INFO")
            subject_metrics["auc"].append(auc)
        except Exception as e:
            # print(f"\nAUC Calculation Error: {e}")
            logger.log("AUC Calculation Error: {}", e, level="WARN")
        empty_line()
        # print(f"Accuracy: {accuracy * 100:.2f}%")
        # print(f"Precision: {precision * 100:.2f}%")
        # print(f"Recall: {recall * 100:.2f}%")
        # print(f"F1 Score: {f1 * 100:.2f}%")
        logger.log("Accuracy: {:.2f}%", accuracy * 100, level="INFO")
        logger.log("Precision: {:.2f}%", precision * 100, level="INFO")
        logger.log("Recall: {:.2f}%", recall * 100, level="INFO")
        logger.log("F1 Score: {:.2f}%", f1 * 100, level="INFO")
        logger.log("\n")
        # empty_line()

        subject_metrics["subject_id"].append(target_subject)
        subject_metrics["accuracy"].append(accuracy)
        subject_metrics["precision"].append(precision)
        subject_metrics["recall"].append(recall)
        subject_metrics["f1"].append(f1)

        torch.cuda.empty_cache()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_confusion_matrix(all_y_true, all_y_pred,
                          "All",
                          os.path.join(log_dir, f"all_{timestamp}"))

    return subject_metrics


if __name__ == "__main__":
    # 参与训练的被试
    # subjects = [
    #     'NP03', 'NP07', 'NP08', 'NP09',
    #     'NP12', 'NP13', 'NP14', 'NP15',
    #     'NP18', 'NP23', 'NP24', 'NP25',
    #     'NP27', 'NP28', 'NP31', 'NP32'
    # ]

    subjects = [
        'NP03', 'NP04', 'NP05', 
        'NP06', 'NP07', 'NP08', 
        'NP09', 'NP10', 'NP11', 
        'NP12', 'NP13', 'NP14', 
        'NP15', 'NP16', 'NP17', 
        'NP18', 'NP19', 'NP20', 
        'NP21', 'NP22', 'NP23', 
        'NP24', 'NP25', 'NP26',
        'NP27', 'NP28', 'NP29', 
        'NP30', 'NP31', 'NP32'
    ]
    # 1, 5, 8
    # 1, 6, 9
    # 生理数据读取
    base_path = 'D:/Data/Group/2-nuclear_data/deeplearning'
    full_df = load_eeg_data(subjects, base_path,
                            1, 5, 9,
                            'basic')
    repeat = 1
    # 请替换为你的 EEG+ECG 特征 CSV 文件路径，需包含 'subject' 和 'label' 列
    log_dir = os.path.join(base_path, 'logs')
    logger = Logger(log_dir)
    performance_list = []
    for r in range(repeat):
        print(f"\n===== Repetition {r + 1} / {repeat} start =====")
        logger.log("info: {}", full_df.shape, level="INFO")
        performance = run_lotso_training(full_df)
        performance_list.append(performance)
        print(f"\n===== Repetition {r + 1} / {repeat} end =======")

    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    logger.save_summary(performance_list, metrics)
