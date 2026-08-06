import os

from evaluation import evaluate_predictions


def test_metrics_generation(tmp_path):
    y_true = [0, 1, 2, 3, 4, 5, 0, 1]
    y_pred = [0, 1, 2, 3, 4, 5, 1, 0]

    # Every output path is redirected into tmp_path. This test used to redirect
    # only the JSON, so its 8-sample toy data overwrote the committed
    # classification report and confusion matrix -- artifacts measured on
    # n=1267 -- with a near-perfect 8-sample result.
    results_dir = tmp_path / "results"
    plots_dir = tmp_path / "plots"
    output_file = results_dir / "evaluation_metrics.json"

    results = evaluate_predictions(
        y_true,
        y_pred,
        output_json=str(output_file),
        results_dir=str(results_dir),
        plots_dir=str(plots_dir),
    )

    assert "metrics" in results
    assert "confusion_matrix" in results
    assert "classification_report" in results

    assert "Macro_F1" in results["metrics"]
    assert "Weighted_F1" in results["metrics"]

    # The side-effect files land beside the JSON, not in the repository.
    assert output_file.exists()
    assert (results_dir / "classification_report.txt").exists()
    assert (plots_dir / "confusion_matrix.png").exists()


def test_evaluate_predictions_does_not_write_into_the_repo(tmp_path):
    """Regression guard for the artifact-clobbering bug.

    Fails if the default paths are ever reintroduced into a caller that means
    to write elsewhere.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tracked = [
        os.path.join(repo_root, "results", "classification_report.txt"),
        os.path.join(repo_root, "plots", "confusion_matrix.png"),
    ]
    before = {p: os.path.getmtime(p) for p in tracked if os.path.exists(p)}

    evaluate_predictions(
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4, 5],
        output_json=str(tmp_path / "m.json"),
        results_dir=str(tmp_path / "r"),
        plots_dir=str(tmp_path / "p"),
    )

    for path, mtime in before.items():
        assert os.path.getmtime(path) == mtime, f"{path} was modified by the test suite"
