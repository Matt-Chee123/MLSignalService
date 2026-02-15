import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import pickle
from training.loader import load_splits
import joblib
from scipy.stats import spearmanr


class FeatureAnalyser:
    def __init__(self, model, feature_names, X_train, y_train, X_test, y_test):
        self.model = model
        self.feature_names = feature_names
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def get_tree_importance(self):
        if not hasattr(self.model, 'feature_importances_'):
            raise AttributeError("Model does not have feature importances attribute")

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
        importance_df['importance_pct'] = 100 * importance_df['importance'] / importance_df['importance'].sum()

        return importance_df.reset_index(drop=True)

    def get_permutation_importance(self, n_repeats=10, random_state=42):
        if self.X_test is None or self.y_test is None:
            raise ValueError("Must have X_test and y_test")

        perm_importance = permutation_importance(self.model, self.X_test, self.y_test, n_repeats=n_repeats, random_state=random_state)

        perm_df = pd.DataFrame({
            'feature_names': self.feature_names,
            'importance_mean': perm_importance.mean(),
            'importance_std': perm_importance.std(),
        }).sort_values('importance_mean', ascending=False)

        return perm_df.reset_index(drop=True)

    def analyse_feature_groups(self):
        importance_df = self.get_tree_importance()
        groups = {
            'log_return': [],
            'rolling_volatility': [],
            'rolling_momentum': [],
            'lagged_returns': [],
            'rsi': [],
            'sma': [],
            'ema': [],
            'macd': []
        }

        for feature in importance_df['feature']:
            for group_name in groups.keys():
                if feature.startswith(group_name):
                    groups[group_name].append(feature)
                    break

        group_stats = {}
        for group_name, features in groups.items():
            if features:
                group_importance = importance_df[importance_df['feature'].isin(features)]
                group_stats[group_name] = {
                    'total_importance': group_importance['importance'].sum(),
                    'mean_importance': group_importance['importance'].mean(),
                    'count': len(features),
                    'top_feature': group_importance.iloc[0]['feature'] if len(group_importance) > 0 else None,
                    'top_importance': group_importance.iloc[0]['importance'] if len(group_importance) > 0 else 0
                }

        return group_stats

    def get_top_features(self, n=20, method='tree'):
        if method == 'tree':
            importance_df = self.get_tree_importance()
        elif method == 'permutation':
            importance_df = self.get_permutation_importance()
        else:
            raise ValueError("Method param is wrong")

        return importance_df.head(n)['feature'].to_list()

    def find_redundant_features(self, correlation_threshold=0.95):
        if self.X_train is None:
            raise ValueError("Need X_train value")

        X_df = pd.DataFrame(self.X_train, columns=self.feature_names)
        corr_matrix = X_df.corr().abs()

        redundant_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1,len(corr_matrix)):
                if corr_matrix.iloc[i, j] > correlation_threshold:
                    redundant_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))
        redundant_pairs.sort(key=lambda x: x[2], reverse=True)
        return redundant_pairs

    def get_features_target_importance(self, target_cumulative=0.95):
        importance_df = self.get_tree_importance()
        features = importance_df[importance_df['cumulative_importance'] <= target_cumulative]['feature'].tolist()

        return features

    def print_importance_report(self, top_n=20):
        print("=" * 70)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 70)

        importance_df = self.get_tree_importance()

        print(f"\n📊 TOP {top_n} FEATURES (Tree Importance)")
        print(f"{'Rank':<6}{'Feature':<35}{'Importance':>12}{'Cumulative':>12}")
        print("-" * 70)

        for idx, row in importance_df.head(top_n).iterrows():
            print(f"{idx + 1:<6}{row['feature']:<35}{row['importance']:>12.6f}{row['cumulative_importance']:>12.2%}")

        print(f"\n📦 FEATURE GROUP ANALYSIS")
        group_stats = self.analyse_feature_groups()

        sorted_groups = sorted(group_stats.items(), key=lambda x: x[1]['total_importance'], reverse=True)

        print(f"{'Group':<25}{'Total Imp':>12}{'Mean Imp':>12}{'Count':>8}")
        print("-" * 70)

        for group_name, stats in sorted_groups:
            print(
                f"{group_name:<25}{stats['total_importance']:>12.6f}{stats['mean_importance']:>12.6f}{stats['count']:>8}")

        print(f"\n🎯 FEATURE REDUCTION OPPORTUNITIES")

        features_95 = self.get_features_target_importance(0.95)
        features_90 = self.get_features_target_importance(0.90)
        features_80 = self.get_features_target_importance(0.80)

        total_features = len(self.feature_names)

        print(f"   80% importance: {len(features_80):>4} features ({len(features_80) / total_features:>6.1%} of total)")
        print(f"   90% importance: {len(features_90):>4} features ({len(features_90) / total_features:>6.1%} of total)")
        print(f"   95% importance: {len(features_95):>4} features ({len(features_95) / total_features:>6.1%} of total)")

        if self.X_train is not None:
            redundant = self.find_redundant_features(correlation_threshold=0.95)

            if redundant:
                print(f"\n⚠️  HIGHLY CORRELATED FEATURES (r > 0.95)")
                print(f"   Found {len(redundant)} pairs of highly correlated features:")

                for feat1, feat2, corr in redundant[:10]:
                    print(f"   • {feat1:<30} <-> {feat2:<30} (r={corr:.3f})")

                if len(redundant) > 10:
                    print(f"   ... and {len(redundant) - 10} more")

        print("=" * 70)


class FeatureSelector:

    def __init__(self, X_train, y_train, X_test, y_test, feature_names: List[str]):

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names

    def forward_selection(self, model_class, max_features: int = 50, metric: str = 'r2') -> List[str]:

        selected_features = []
        remaining_features = self.feature_names.copy()

        best_score = -np.inf if metric in ['r2', 'rank_ic'] else np.inf

        for _ in range(min(max_features, len(self.feature_names))):
            scores = []

            for feature in remaining_features:
                test_features = selected_features + [feature]
                test_indices = [self.feature_names.index(f) for f in test_features]

                X_train_subset = self.X_train[:, test_indices]
                X_test_subset = self.X_test[:, test_indices]

                model = model_class()
                model.fit(X_train_subset, self.y_train)

                score = self._evaluate_model(model, X_test_subset, self.y_test, metric)
                scores.append(score)

            if metric in ['r2', 'rank_ic']:
                best_idx = np.argmax(scores)
                new_score = scores[best_idx]
                improved = new_score > best_score
            else:
                best_idx = np.argmin(scores)
                new_score = scores[best_idx]
                improved = new_score < best_score

            if improved:
                best_feature = remaining_features[best_idx]
                selected_features.append(best_feature)
                remaining_features.remove(best_feature)
                best_score = new_score

                print(f"Added {best_feature}: {metric} = {new_score:.4f}")
            else:
                print(f"No improvement. Stopping at {len(selected_features)} features.")
                break

        return selected_features

    def backward_elimination(self, model_class, min_features: int = 10, metric: str = 'r2') -> List[str]:

        remaining_features = self.feature_names.copy()

        model = model_class()
        model.fit(self.X_train, self.y_train)
        best_score = self._evaluate_model(model, self.X_test, self.y_test, metric)
        print("ehre")
        while len(remaining_features) > min_features:
            scores = []

            for feature in remaining_features:
                test_features = [f for f in remaining_features if f != feature]
                test_indices = [self.feature_names.index(f) for f in test_features]

                X_train_subset = self.X_train[:, test_indices]
                X_test_subset = self.X_test[:, test_indices]

                model = model_class()
                model.fit(X_train_subset, self.y_train)

                score = self._evaluate_model(model, X_test_subset, self.y_test, metric)
                scores.append((feature, score))

            if metric in ['r2', 'rank_ic']:
                best_removal = max(scores, key=lambda x: x[1])
                improved = best_removal[1] >= best_score
            else:
                best_removal = min(scores, key=lambda x: x[1])
                improved = best_removal[1] <= best_score

            if improved or best_removal[1] >= best_score * 0.99:
                removed_feature = best_removal[0]
                remaining_features.remove(removed_feature)
                best_score = best_removal[1]

                print(f"Removed {removed_feature}: {metric} = {best_score:.4f}")
            else:
                print("Cannot remove more features without significant degradation.")
                break

        return remaining_features

    def _evaluate_model(self, model, X, y, metric: str) -> float:

        predictions = model.predict(X)

        if metric == 'r2':
            return model.score(X, y)
        elif metric == 'mse':
            return np.mean((predictions - y) ** 2)
        elif metric == 'rank_ic':
            return spearmanr(predictions, y)[0]
        else:
            raise ValueError(f"Unknown metric: {metric}")

model = joblib.load('../training/artifacts/rf_signal_v1/20260213_175247/models/model.pkl')
splits = load_splits('../data/datasets/run_20260212_210236')

for idx, (train, test) in enumerate(splits):
    X_train, y_train = train.drop(columns='label'), train['label']
    X_test, y_test = test.drop(columns='label'), test['label']

    print("Run: ", idx)
    selector = FeatureSelector(
        X_train=X_train.values,
        y_train=y_train.values,
        X_test=X_test.values,
        y_test=y_test.values,
        feature_names=X_train.columns.tolist()
    )
    selected_features = selector.backward_elimination(
        lambda: RandomForestRegressor(n_estimators=200, random_state=42),
        min_features=20,
        metric="rank_ic"
    )

    print(selected_features)


    print(f"\n=== Split {idx} ===")
    analyser.print_importance_report(top_n=10)