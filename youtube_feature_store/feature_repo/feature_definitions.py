from datetime import timedelta

from feast.types import Int64
from feast import Entity, FeatureView, Field, FileSource, Project
from feast.types import Int64, Float32

project = Project(
    name="youtube_feature_store",
    description="YouTube Comment Sentiment Features"
)


comment = Entity(
    name="comment",
    join_keys=["comment_id"],
    value_type=Int64
)

comment_source = FileSource(
    name="comment_source",
    path="data/comments.parquet",
    timestamp_field="event_timestamp",
)

comment_features = FeatureView(
    name="comment_features",
    entities=[comment],
    ttl=timedelta(days=1),
    schema=[
        Field(name="comment_length", dtype=Int64),
        Field(name="word_count", dtype=Int64),
        Field(name="uppercase_ratio", dtype=Float32),
        Field(name="exclamation_count", dtype=Int64),
    ],
    online=True,
    source=comment_source,
)