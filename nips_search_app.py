import streamlit as st
from PaperProcessing import load_nips_abstracts, get_tagged_documents, process_abstract
from TextAnalysis import VectorAnalysis, get_doc2vec_model, get_n_similar
import numpy as np
from gensim.models import Doc2Vec
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

avail_years = [2017, 2018, 2019, 2020, 2021]

if 'nips_dct' not in st.session_state:
    st.session_state['search_term'] = ''
    raw_abstracts = load_nips_abstracts()
    st.session_state['nips_dct'] = raw_abstracts
    st.session_state['year'] = 0
    year_model_dct = {}
    for year in avail_years:
        model = Doc2Vec.load(f"Doc2VecModels/{year}_model")
        year_model_dct[year] = model
    st.session_state['model'] = year_model_dct


def get_scatter_plot(df_2d, clusters_show):
    df_to_show = df_2d[df_2d['Cluster Idx'].isin(clusters_show)]
    if len(clusters_show) == 1:
        scaler = MinMaxScaler(feature_range=(0, 5))
        scaled_values = scaler.fit_transform(df_to_show['Distance to Centroid'].values.reshape((-1, 1)))
        df_to_show['Scaled Dist'] = scaled_values.reshape(-1, 1)
        return px.scatter(df_to_show, x='X', y='Y',
                          size='Similarity Score', color='Scaled Dist', hover_name='Titles',
                          hover_data=['Similarity Score'])
    else:
        return px.scatter(df_to_show, x='X', y='Y',
                          size='Similarity Score', color='Cluster Idx', hover_name='Titles',
                          hover_data=['Similarity Score'])


def get_hist(df_2d, clusters_show):
    df_to_show = df_2d[df_2d['Cluster Idx'].isin(clusters_show)]
    return px.histogram(df_to_show, x='Similarity Score')


def get_pie(df_clustering_summary):
    return px.pie(df_clustering_summary, values='Number of Docs', names='Group')


def update_visuals(df_2d, clusters_show, scatter_container, hist_container):
    with scatter_container:
        scatter = get_scatter_plot(df_2d, clusters_show)
        st.plotly_chart(scatter)

    with hist_container:
        hist = get_hist(df_2d, clusters_show)
        st.plotly_chart(hist)
    return




search_submitted = False
with st.form("Conference Search"):
    year = st.selectbox(label='Year for Search', options=avail_years)
    search_term_pos = st.text_input(label='Positive Search Terms')
    search_term_neg = st.text_input(label='Negative Search Terms')
    search_submitted = st.form_submit_button("Run Search")





if search_submitted:
    st.session_state['search_term'] = search_term_pos
    model = st.session_state['model'][year]
    n_sim = get_n_similar(model, search_term_pos, search_term_neg, 200)

    with st.spinner(f"Running Document Analysis for \"{search_term_pos}\" "):
        raw_abstracts = st.session_state['nips_dct'][year]
        vec_analysis = VectorAnalysis(n_sim, model, raw_abstracts)
        df_2d, df_clustering_summary = vec_analysis.get_results()

    pie = get_pie(df_clustering_summary)
    st.subheader("Composition of Most Relevant Articles by Cluster")
    st.plotly_chart(pie)

    st.header(f"Results for \"{search_term_pos}\"")
    st.session_state['vec_analysis'] = vec_analysis
    st.session_state['df_2d'] = df_2d
    st.session_state['df_clustering_summary'] = df_clustering_summary

    num_clusters = vec_analysis.k_means.cluster_centers_.shape[0]
    cluster_tab_labels = [f"Cluster {idx}" for idx in range(num_clusters)]
    cluster_tab_labels.append("All Clusters")
    cluster_tabs = st.tabs(cluster_tab_labels)

    for i in range(num_clusters):
        with cluster_tabs[i] as container:
            to_show = {i}

            scatter_plot = get_scatter_plot(df_2d, to_show)
            st.subheader(f"2d Visualization of Cluster {i}")
            st.plotly_chart(scatter_plot, use_container_width=False)

            hist = get_hist(df_2d, to_show)
            st.subheader(f"Distribution of Similarity Scores for Cluster {i}")
            st.plotly_chart(hist, use_container_width=False)

    with cluster_tabs[-1]:
        to_show = {i for i in range(num_clusters)}
        scatter_plot = get_scatter_plot(df_2d, to_show)
        st.subheader("2d Visualization for All Clusters")
        st.plotly_chart(scatter_plot, use_container_width=False)

        hist = get_hist(df_2d, to_show)
        st.subheader("Distribution of Similarity Scores for All Clusters")
        st.plotly_chart(hist, use_container_width=False)



    with st.sidebar:
        st.header("NIPS Papers by Cluster")
        for i in range(num_clusters):
            df_to_show = df_2d[df_2d['Cluster Idx'] == i]
            with st.expander(f"Papers in Cluster {i}"):
                for idx in range(len(df_to_show.index)):
                    st.header(f"\"{df_to_show.iloc[idx]['Titles']}\"")
                    st.write(df_to_show.iloc[idx]['Abstracts'])
