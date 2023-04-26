#-*- coding: utf-8 -*-
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pickle
import matplotlib.pyplot as plt


with open(r'C:\Users\admin\Desktop\Recommendation System\location_Model.pkl','rb') as file:
    tfidf = pickle.load(file)

DATA = pd.read_csv(r'C:\Users\admin\Desktop\Recommendation System\Clean_data.csv')


# Pie-chart for Job type
st.subheader("Pie-chart for Job_type")
jobtype_count = DATA['Job_type'].value_counts()
labels_list = ['On-site','Hybrid','Remote'] 
fig1,ax1 = plt.subplots()
ax1.pie(jobtype_count , labels = labels_list , autopct ='"%0.2f%%"')
ax1.axis('equal')
st.pyplot(fig1)

# Pie-chart for Involvement
st.subheader("Pie chart for Involvement")
invo_count = DATA['Involvement'].value_counts()
invo = invo_count.head(5)
invo_lab=list(invo.index)
labels_list = invo_lab
fig1,ax1 = plt.subplots()
plt.pie(invo , labels = labels_list , autopct ='"%0.2f%%"')
ax1.axis('equal')
st.pyplot(fig1)

# pie chart for Industry
st.subheader("Pie chart for Industry")
industry_count = DATA['Industry'].value_counts()
ind = industry_count.head(5)
ind_lab = list(ind.index)
labels_list = ind_lab
fig1,ax1 = plt.subplots()
plt.pie(ind , labels = labels_list , autopct ='"%0.2f%%"')
ax1.axis('equal')
st.pyplot(fig1)

# Pie chart for Company
st.subheader("Pie chart for company")
company_count = DATA['Company'].value_counts()
com=company_count.head(5)
com_lab=list(com.index)
labels_list = com_lab
fig1,ax1 = plt.subplots()
plt.pie(com , labels = labels_list , autopct ='"%0.2f%%"')
ax1.axis('equal')
st.pyplot(fig1)

# Pie chart for Job_Name
st.subheader("Pie chart for Job Name")
jobname_count = DATA['Job_Name'].value_counts()
job=jobname_count.head(5)
job_lab=list(job.index)
labels_list = job_lab
fig1,ax1 = plt.subplots()
plt.pie(job , labels = labels_list , autopct ='"%0.2f%%"')
ax1.axis('equal')
st.pyplot(fig1)

#Pie chart for Location
st.subheader("Pie chart for Location")
loc_count = DATA['Location'].value_counts()
loc = loc_count.head(8)
loc_lab = list(loc.index)
labels_list = loc_lab
fig1,ax1 = plt.subplots()
plt.pie(loc , labels = labels_list , autopct ='"%0.2f%%"')
ax1.axis('equal')
st.pyplot(fig1)
 

st.title('Model Deployment: Location Bassed Job Recommendation')
jobname = st.text_input('Enter job name: ', 'Data Analyst')
st.write('Job name is ', jobname)


tfidf_matrix = tfidf.fit_transform(DATA["Location"])

cosine_sim = linear_kernel(tfidf_matrix,tfidf_matrix)
print(cosine_sim)

def get_recommendations(title, cosine_sim = cosine_sim):
    try:
        ind_job =  pd.Series(DATA["Job_Name"]).drop_duplicates()
        indices1 = pd.Series(ind_job.index,index=ind_job.values)
        idx = indices1[title]
        print(idx)
        sim_scores = enumerate(cosine_sim[idx])       
        sim_scores = sorted(sim_scores,key=lambda x:x[1],reverse =True)
        sim_scores = sim_scores[1:11] 
        
        sim_index = [i[0] for i in sim_scores]
        print('recommendation') 
        print(DATA["Job_Name"].iloc[(sim_index)])
        recommended_job = DATA["Job_Name"].iloc[(sim_index)]
    except Exception:
        recommended_job = []

    return recommended_job


st.subheader("Top 10 recommendations based on location of job input are :")
recommend_job = get_recommendations(jobname)
if len(recommend_job)>0:
    st.write(recommend_job)
else:
    st.write("No jobs recommended for input job")







