import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

reference_data = pd.DataFrame({
    "comment_length": [20, 25, 30, 35, 40],
    "word_count": [5, 6, 7, 8, 9]
})

current_data = pd.DataFrame({
    "comment_length": [50, 55, 60, 65, 70],
    "word_count": [10, 11, 12, 13, 14]
})

report = Report(
    metrics=[DataDriftPreset()]
)

report.run(
    reference_data=reference_data,
    current_data=current_data
)

print(report.dict())