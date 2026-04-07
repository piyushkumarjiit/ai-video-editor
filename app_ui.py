"""
FILE: app_ui.py
ROLE: Interactive Redaction Controller (Streamlit)
-------------------------------------------------------------------------
DESCRIPTION:
The frontend for the pipeline. It reads 'ui_manifest.json' and presents 
a checkbox list of all detected entities. Users select who to redact, 
and the script generates the final 'redaction_manifest.json' for the 
rendering engine.
-------------------------------------------------------------------------
"""

import streamlit as st
import json
import os

def main():
    st.set_page_config(page_title="AI Video Redactor", layout="wide")
    st.title("Entity Selection for Redaction")

    manifest_path = "ui_manifest.json"
    if not os.path.exists(manifest_path):
        st.warning(f"Manifest {manifest_path} not found. Run the detection pipeline first.")
        return

    with open(manifest_path, 'r') as f:
        data = json.load(f)

    st.sidebar.header("Detected Entities")
    selected_entities = []

    # Parse unique entities from the manifest
    entities = {}
    for frame, details in data.get("frames", {}).items():
        for entity in details.get("entities", []):
            if entity["id"] not in entities:
                entities[entity["id"]] = entity["label"]

    for ent_id, label in entities.items():
        if st.sidebar.checkbox(f"Redact {label} ({ent_id})", value=False):
            selected_entities.append(ent_id)

    if st.sidebar.button("Generate Redaction Manifest"):
        output = {"selected_for_redaction": selected_entities}
        with open("redaction_manifest.json", "w") as f:
            json.dump(output, f, indent=4)
        st.sidebar.success("redaction_manifest.json saved! Ready for apply_redaction.py")

if __name__ == "__main__":
    main()