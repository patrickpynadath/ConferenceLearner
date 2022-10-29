# ConferenceLearner

This is a quick web application I made to make my own research a bit easier. I've always been stumped about where to start when picking what paper's to read, 
and I wanted to see if there was a way to use ML to at least point me in the right direction. Given key terms that are similar to the target topic, and 
key terms that are dissimilar from the target topic, the app reads through all the abstracts for a given NIPS proceeding and uses the vector representations 
of both the terms and abstracts to find and organize the relevant articles. As of right now, the app uses a Doc2Vec model trained on a specific year's proceedings
to vectorize the abstracts. From there, it uses PCA to represent the abstracts in a lower dimensional space, from which k-means clustering is used 
to group the relevant articles. I use plotly and streamlit for everything related to the front end and data visualization. 

This is still a work in progress and will get less buggy and more useful as time goes on. 
