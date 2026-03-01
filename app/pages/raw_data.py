import streamlit as st
import pandas as pd
import json
from pathlib import Path
from utils.data_loader import list_experiment, list_runs, get_run_metadata
from config import ARTIFACTS_DIR


def render():
    st.header("🗂️ Raw Data Explorer")

    st.write("Browse and inspect raw experiment artifacts for debugging or custom analysis.")

    experiments = list_experiment()
    if not experiments:
        st.warning("No experiments found")
        return

    experiment = st.sidebar.selectbox(
        "Experiment",
        experiments,
        format_func=lambda x: x.name if hasattr(x, 'name') else str(x)
    )

    runs = list_runs(experiment)
    if not runs:
        st.warning(f"No runs found for {experiment.name}")
        return

    run_id = st.sidebar.selectbox(
        "Run",
        runs,
        format_func=lambda x: x.name if hasattr(x, 'name') else str(x)
    )

    metadata = get_run_metadata(run_id)

    st.subheader("📋 Run Metadata")

    col1, col2, col3 = st.columns(3)
    col1.metric("Experiment", metadata['experiment_name'])
    col2.metric("Run ID", metadata['run_id'])
    col3.metric("Date", metadata['date'])

    st.subheader("📁 Directory Structure")

    def get_directory_tree(path, prefix="", max_depth=3, current_depth=0):
        if current_depth >= max_depth:
            return []

        items = []
        try:
            contents = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))

            for item in contents:
                if item.name.startswith('.'):
                    continue

                is_last = item == contents[-1]
                connector = "└── " if is_last else "├── "

                if item.is_dir():
                    items.append(f"{prefix}{connector}📁 {item.name}/")
                    extension = "    " if is_last else "│   "
                    items.extend(get_directory_tree(item, prefix + extension, max_depth, current_depth + 1))
                else:
                    size_mb = item.stat().st_size / (1024 * 1024)
                    size_str = f"{size_mb:.2f} MB" if size_mb > 0.01 else f"{item.stat().st_size} bytes"
                    items.append(f"{prefix}{connector}📄 {item.name} ({size_str})")
        except PermissionError:
            items.append(f"{prefix}[Permission Denied]")

        return items

    tree = get_directory_tree(run_id)
    st.code("\n".join(tree), language="")

    st.subheader("🔍 File Browser")

    all_files = []
    for file_path in run_id.rglob("*"):
        if file_path.is_file() and not file_path.name.startswith('.'):
            rel_path = file_path.relative_to(run_id)
            all_files.append((str(rel_path), file_path))

    if not all_files:
        st.info("No files found in this run")
        return

    file_groups = {}
    for rel_path, full_path in all_files:
        dir_name = Path(rel_path).parent
        if str(dir_name) == '.':
            dir_name = 'Root'
        else:
            dir_name = str(dir_name)

        if dir_name not in file_groups:
            file_groups[dir_name] = []
        file_groups[dir_name].append((rel_path, full_path))

    selected_group = st.selectbox("Select Directory", sorted(file_groups.keys()))

    if selected_group:
        files_in_group = file_groups[selected_group]
        selected_file_rel = st.selectbox(
            "Select File",
            [rel for rel, _ in files_in_group]
        )

        selected_file_path = next(
            full for rel, full in files_in_group if rel == selected_file_rel
        )

        st.write(f"**File:** `{selected_file_rel}`")
        st.write(f"**Size:** {selected_file_path.stat().st_size / 1024:.2f} KB")

        suffix = selected_file_path.suffix.lower()

        if suffix == '.json':
            st.subheader("📄 JSON Content")

            try:
                with open(selected_file_path) as f:
                    data = json.load(f)

                st.json(data)

                json_str = json.dumps(data, indent=2)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_str,
                    file_name=selected_file_path.name,
                    mime="application/json"
                )

            except Exception as e:
                st.error(f"Error loading JSON: {e}")

        elif suffix == '.csv':
            st.subheader("📊 CSV Content")

            try:
                df = pd.read_csv(selected_file_path)

                st.write(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")

                st.dataframe(df.head(50), use_container_width=True)

                with st.expander("Column Information"):
                    col_info = pd.DataFrame({
                        'Column': df.columns,
                        'Type': df.dtypes.values,
                        'Non-Null': df.count().values,
                        'Null %': ((df.isnull().sum() / len(df)) * 100).values
                    })
                    st.dataframe(col_info)

                csv_str = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_str,
                    file_name=selected_file_path.name,
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Error loading CSV: {e}")

        elif suffix == '.parquet':
            st.subheader("📊 Parquet Content")

            try:
                df = pd.read_parquet(selected_file_path)

                st.write(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")

                st.dataframe(df.head(50), use_container_width=True)

                with st.expander("Column Information"):
                    col_info = pd.DataFrame({
                        'Column': df.columns,
                        'Type': df.dtypes.values,
                        'Non-Null': df.count().values,
                        'Null %': ((df.isnull().sum() / len(df)) * 100).values
                    })
                    st.dataframe(col_info)

                csv_str = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv_str,
                    file_name=selected_file_path.stem + ".csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Error loading Parquet: {e}")

        elif suffix == '.log':
            st.subheader("📝 Log Content")

            try:
                with open(selected_file_path) as f:
                    log_content = f.read()

                num_lines = st.slider("Number of lines to show", 10, 1000, 100, 10)
                lines = log_content.split('\n')

                st.code('\n'.join(lines[-num_lines:]), language="")

                st.download_button(
                    label="📥 Download Log",
                    data=log_content,
                    file_name=selected_file_path.name,
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"Error loading log: {e}")

        elif suffix in ['.png', '.jpg', '.jpeg']:
            st.subheader("🖼️ Image")

            try:
                st.image(str(selected_file_path), use_column_width=True)

                with open(selected_file_path, 'rb') as f:
                    st.download_button(
                        label="📥 Download Image",
                        data=f,
                        file_name=selected_file_path.name,
                        mime=f"image/{suffix[1:]}"
                    )
            except Exception as e:
                st.error(f"Error loading image: {e}")

        elif suffix in ['.pkl', '.pickle']:
            st.subheader("🥒 Pickle File")

            st.warning("Pickle files contain binary data and cannot be previewed directly.")
            st.info(f"File size: {selected_file_path.stat().st_size / (1024 * 1024):.2f} MB")

            with open(selected_file_path, 'rb') as f:
                st.download_button(
                    label="📥 Download Pickle",
                    data=f,
                    file_name=selected_file_path.name,
                    mime="application/octet-stream"
                )

        elif suffix in ['.txt', '.md', '.py', '.yaml', '.yml']:
            st.subheader("📄 Text Content")

            try:
                with open(selected_file_path) as f:
                    content = f.read()

                language_map = {
                    '.py': 'python',
                    '.yaml': 'yaml',
                    '.yml': 'yaml',
                    '.md': 'markdown',
                    '.txt': ''
                }

                st.code(content, language=language_map.get(suffix, ''))

                st.download_button(
                    label="📥 Download File",
                    data=content,
                    file_name=selected_file_path.name,
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"Error loading text file: {e}")

        else:
            st.info(f"Preview not available for {suffix} files")

            with open(selected_file_path, 'rb') as f:
                st.download_button(
                    label="📥 Download File",
                    data=f,
                    file_name=selected_file_path.name,
                    mime="application/octet-stream"
                )