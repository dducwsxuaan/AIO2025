import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from spam_classifier import SpamClassifierPipeline
from config import SpamClassifierConfig

# --- Page config ---
st.set_page_config(page_title="Spam Mail Dashboard", layout="centered")

# --- Load pipeline/model once ---
@st.cache_resource
def load_pipeline():
    config = SpamClassifierConfig()
    pipeline = SpamClassifierPipeline(config)
    pipeline.train()  # Loads from cache if available
    return pipeline

pipeline = load_pipeline()

# --- Sidebar navigation ---
st.sidebar.title("📧 Spam Mail Dashboard")
page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Overview",
        "✉️ Single Message",
        "📂 Batch Classification",
        "📊 Data Analysis",
        "🧠 Embedding Visualization",
        "📈 Model Performance",
        "🔍 Similarity Search",
        "🗃️ Vector DB Analysis"
    ]
)

# --- Data loading utility ---
@st.cache_data
def load_sample_data():
    # You can replace this with your real dataset
    df = pd.read_csv('dataset/2cls_spam_text_cls.csv')
    return df

if "df" not in st.session_state:
    st.session_state["df"] = load_sample_data()

# --- Overview ---
if page == "🏠 Overview":
    st.title("📧 Spam/Ham Mail Classifier")
    st.write("""
    Welcome to the Spam/Ham Mail Classifier Dashboard!

    **Instructions:**
    - Use the sidebar to navigate between different features of the app.
    - **Single Message:** Classify a single email/message as spam or ham.
    - **Batch Classification:** Upload a CSV file and classify multiple messages at once.
    - **Data Analysis:** Explore your dataset with charts and statistics.
    - **Embedding Visualization:** Visualize message embeddings in 2D/3D.
    - **Model Performance:** View model metrics and confusion matrix.
    - **Similarity Search:** Find messages similar to your input.
    - **Vector DB Analysis:** Analyze the vector database and clustering.

    Please select a feature from the sidebar to get started.
    """)
# --- Single Message Classification ---
elif page == "✉️ Single Message":
    st.header("Classify a Single Message")
    msg = st.text_area("Enter your message:")
    k_value = st.slider("Number of Nearest Neighbors (k)", 1, 10, 3)
    if st.button("Classify"):
        result = pipeline.predict(msg, k=k_value)
        pred = result['prediction']
        st.success(f"Prediction: **{pred.upper()}**")
        if "neighbors" in result:
            st.markdown("**Top Nearest Neighbors:**")
            for i, neighbor in enumerate(result["neighbors"]):
                st.write(f"{i+1}. [{neighbor['label'].upper()}] {neighbor['message'][:100]}... (score: {neighbor['score']:.3f})")

# --- Batch Classification ---
elif page == "📂 Batch Classification":
    st.header("Batch Classification")
    uploaded = st.file_uploader("Upload CSV for batch classification", type="csv")
    k_value = st.slider("Number of Nearest Neighbors (k)", 1, 10, 3, key="batch_k")
    if uploaded:
        df_batch = pd.read_csv(uploaded)
        if "Message" not in df_batch.columns:
            st.error("CSV must have a 'Message' column.")
        else:
            with st.spinner("Classifying..."):
                df_batch["Prediction"] = df_batch["Message"].apply(lambda x: pipeline.predict(x, k=k_value)['prediction'])
            st.dataframe(df_batch)
            st.download_button("Download Results", df_batch.to_csv(index=False), "results.csv")

# --- Data Analysis ---
elif page == "📊 Data Analysis":
    st.header("Exploratory Data Analysis")
    df = st.session_state["df"]
    st.subheader("Label Distribution")
    fig = px.pie(df, names="Label", title="Spam vs Ham Distribution")
    st.plotly_chart(fig)
    st.subheader("Message Length Distribution")
    df["Length"] = df["Message"].str.len()
    fig2 = px.histogram(df, x="Length", color="Label", nbins=20)
    st.plotly_chart(fig2)

# --- Embedding Visualization ---
elif page == "🧠 Embedding Visualization":
    st.header("Embedding Visualization")
    st.write("""
    Visualize how your messages are clustered in the embedding space.
    - Each point represents a message.
    - Colors show spam/ham labels.
    - Use this to check if your model separates spam and ham well!
    """)
    # Example: Generate fake embeddings for demo
    df = st.session_state["df"]
    np.random.seed(42)
    embeddings = np.random.randn(len(df), 16)  # Replace with your real embeddings

    method = st.selectbox("Dimensionality Reduction Method", ["PCA", "t-SNE"])
    n_points = st.slider("Number of points to plot", 100, min(1000, len(df)), min(len(df), 500), step=100)

    # Reduce to 2D
    if method == "PCA":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
    else:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
    reduced = reducer.fit_transform(embeddings[:n_points])

    viz_df = pd.DataFrame({
        "x": reduced[:, 0],
        "y": reduced[:, 1],
        "Label": df["Label"][:n_points],
        "Message": df["Message"][:n_points]
    })

    import plotly.express as px
    fig = px.scatter(
        viz_df, x="x", y="y", color="Label",
        hover_data=["Message"],
        title=f"{method} Embedding Visualization"
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Model Performance ---
elif page == "📈 Model Performance":
    st.header("Model Performance")
    st.write("Evaluate your model with standard classification metrics.")

    # Example: Use your pipeline or sklearn for evaluation
    df = st.session_state["df"]
    y_true = df["Label"]
    # For demo, use predictions from pipeline (replace with your real predictions)
    y_pred = df["Message"].apply(lambda x: pipeline.predict(x, k=3)['prediction'])

    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    st.subheader("Classification Report")
    report = classification_report(y_true, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred, labels=["ham", "spam"])
    fig_cm, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["ham", "spam"], yticklabels=["ham", "spam"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig_cm)

# --- Similarity Search ---
elif page == "🔍 Similarity Search":
    st.header("Similarity Search")
    query = st.text_area("Enter a message to find similar emails:")
    k_value = st.slider("Number of Nearest Neighbors (k)", 1, 10, 3, key="sim_k")
    if st.button("Find Similar"):
        result = pipeline.predict(query, k=k_value)
        st.markdown("**Top Similar Messages:**")
        for i, neighbor in enumerate(result.get("neighbors", [])):
            st.write(f"{i+1}. [{neighbor['label'].upper()}] {neighbor['message'][:100]}... (score: {neighbor['score']:.3f})")

# --- Vector DB Analysis ---
elif page == "🗃️ Vector DB Analysis":
    st.header("Vector Database Analysis")
    st.info("Show cosine similarity heatmap, clustering, and vector stats here.")

# --- Footer ---
st.markdown("---")
st.caption("Built with Streamlit | Replace dummy logic with your model and data pipeline.")