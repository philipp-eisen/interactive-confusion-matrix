from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
import altair as alt

from icm.models import ConfusionMatrixRequest, ENormalize


def cm_chart(req: ConfusionMatrixRequest):
    cm = confusion_matrix(req.actual, req.predicted)
    cm_norm_pred = normalize(cm, axis=0)
    cm_norm_actu = normalize(cm, axis=1)

    predicted_id, actual_id = np.meshgrid(range(len(cm[0])), range(len(cm[1])))
    chart_records = {
        "predicted": predicted_id.ravel(),
        "actual": actual_id.ravel(),
        "count": cm.ravel(),
        "frac-pred": cm_norm_pred.ravel(),
        "frac-actual": cm_norm_actu.ravel(),
    }

    if req.class_names:
        chart_records = {
            **chart_records,
            "name-actual": [x[0] for x in product(req.class_names, req.class_names)],
            "name-pred": [x[1] for x in product(req.class_names, req.class_names)],
        }

    if req.normalize == ENormalize.PREDICTED:
        color = "frac-pred"
    elif req.normalize == ENormalize.ACTUAL:
        color = "frac-actual"
    elif req.normalize == ENormalize.NO:
        color = "actual"
    else:
        raise NotImplementedError

    properties = {"width": 600, "height": 600}
    if req.title:
        properties["title"] = req.title

    cm_df = pd.DataFrame(chart_records)
    selected = alt.selection(type="single", fields=["predicted", "actual"])
    conf_matrix_plot = (
        alt.Chart(cm_df)
        .mark_rect()
        .encode(
            x="predicted:O",
            y="actual:O",
            color=alt.Color(color, scale=alt.Scale(scheme="bluepurple")),
            tooltip=list(cm_df.columns),
        )
        .properties(**properties)
        .add_selection(selected)
    )
    if req.features:
        pred_df = pd.DataFrame(
            {"feature": req.features, "actual": req.actual, "predicted": req.predicted,}
        )
        ranked_text = (
            alt.Chart(pred_df)
            .mark_text(align="left")
            .encode(y=alt.Y("row_number:O", axis=None))
            .transform_window(row_number="row_number()")
            .transform_filter(selected)
            .transform_window(rank="rank(row_number)")
            .transform_filter(alt.datum.rank < 30)
        )
        examples = ranked_text.encode(text="feature:N")
        return conf_matrix_plot | examples
    return conf_matrix_plot
