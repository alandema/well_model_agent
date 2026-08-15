#!/bin/sh

python -m app.well_model.multi_well_model --wells 4 --interval 1.0 &
streamlit run app/main.py --server.address 0.0.0.0 --server.port 8601
