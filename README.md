# 📊 [Twitter Analytics Dashboard](https://twitter-analytics-dashboard.streamlit.app/)

A comprehensive Streamlit application for analyzing your Twitter performance with beautiful visualizations and actionable insights.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🚀 Features

### 📈 **Analytics & Insights**
- **Post Performance Comparison** - Compare your top posts with interactive grouped bar charts
- **Time-based Analysis** - Discover optimal posting times and days with rainbow visualizations
- **Engagement Analysis** - Track likes, reposts, replies, and bookmarks over time
- **Sentiment Analysis** - Analyze content sentiment with word clouds and performance correlations
- **Viral Potential Analysis** - Identify what makes your content go viral

### 🎨 **Visual Features**
- **Interactive Charts** - Powered by Plotly with hover details and zoom capabilities
- **Word Cloud Generation** - Beautiful word frequency visualizations
- **Multi-dimensional Bubble Charts** - Size, color, and position encoding for rich insights
- **Rainbow Color Palettes** - Engaging and accessible color schemes
- **Responsive Design** - Works on desktop and mobile devices

### 🔍 **Advanced Analysis**
- **Content Length Optimization** - Find the perfect tweet length for maximum engagement
- **Hashtag Performance** - Multi-dimensional analysis of hashtag effectiveness
- **Mention Impact** - Understand which accounts drive the most impressions
- **Viral vs Non-Viral Comparison** - Normalized metrics comparison with percentage differences

## 📋 Requirements

### **Twitter Account Requirements**
- **Twitter Premium/Professional Account**
- Access to Twitter Analytics dashboard
- Historical tweet data (minimum 30 days recommended)

## 📊 Data Requirements

### **Required Files**
Upload two CSV files to the dashboard:

1. **`account_analytics_content_*.csv`** (Tweet-level data)
   - Individual tweet performance metrics
   - Columns: Date, Tweet text, Impressions, Likes, Engagements, etc.

2. **`account_overview_analytics_*.csv`** (Daily summary data)
   - Daily aggregated account performance
   - Columns: Date, Total impressions, Total engagements, Followers, etc.

### **Supported Languages**
- **Portuguese:** `Data`, `Texto do post`, `Impressões`, `Curtidas`, `Engajamentos`
- **English:** `Date`, `Post text`, `Impressions`, `Likes`, `Engagements`

## 📱 How to Get Your Twitter Data

### **Step 1: After upgrading to Professional Access Analytics**
1. Go to **Premium** section in sidebar
2. Click **Analytics**

![image](https://github.com/user-attachments/assets/67e9f888-4760-47b9-8ece-7538454b82d6)
![image](https://github.com/user-attachments/assets/87d34afb-dbe5-4e01-b729-6bbb3809f8be)

### **Step 2: Download Overview Data**
1. Go to **Overview** tab
2. Select **1 Year** time period
3. Click the **download button** (📥)

![image](https://github.com/user-attachments/assets/c01675f2-0b7d-4a62-adf4-3dfa4d9a88a6)

### **Step 3: Download Content Data**
1. Go to **Content** tab
2. Select **1 Year** time period
3. Click the **download button** (📥)

![image](https://github.com/user-attachments/assets/f4d10f16-7a3f-4499-8193-0e3e4805b10d)

## 🎯 Dashboard Sections

### **1. Account Overview**

![image](https://github.com/user-attachments/assets/8bdf8bf5-8319-4796-a920-84cb2894fbed)

- Key performance metrics cards
- Follower growth visualization
- Correlation heatmap between metrics

### **2. Post Performance Comparison**

![image](https://github.com/user-attachments/assets/402fbaea-e6e5-4c5a-99da-25a45d319f15)

- Interactive grouped bar charts
- Customizable sorting by any metric
- Detailed hover information with content preview
- CSV export functionality

### **3. Engagement Analysis**

![image](https://github.com/user-attachments/assets/72711db7-905c-4c23-8b15-f027f723ea84)

- Time series with separate subplots for each metric
- Engagement rate distribution analysis
- Daily aggregated performance tracking

### **4. Time-based Performance**
- Rainbow-colored day-of-week analysis
- Optimal posting time recommendations
- Performance annotations and insights

### **5. Content Analysis**

![image](https://github.com/user-attachments/assets/c6ff4be8-1fd6-4ec9-a6c3-ca75e129c2ee)

- Hashtag performance matrix (multi-dimensional)
- Mention impact analysis with frequency encoding
- Tweet length optimization with outlier handling


### **6. Sentiment Analysis**

![image](https://github.com/user-attachments/assets/8794ab28-7db9-4cdf-b76d-49ce3b28f818)

- Real-time word cloud generation
- Sentiment evolution over time
- Performance correlation by sentiment type
- Word frequency analysis with stop-word filtering

### **7. Viral Potential Analysis**

![image](https://github.com/user-attachments/assets/bb57163c-d2b0-4adc-8b71-b233b6158c8e)

- Viral threshold identification (top 10% impressions)
- Normalized characteristic comparison
- Timing analysis for viral content
- Actionable insights and recommendations

## 🤝 Contributing

### **How to Contribute**

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Submit a pull request with detailed description
5. Send me an email on felipe.g.datascience@gmail.com or a DM on my LinkedIn Profile at https://www.linkedin.com/in/felipe-gabriel0/

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### **Common Issues**
- **File upload errors**: Ensure CSV files are properly formatted UTF-8
- **Memory issues**: Reduce dataset size or increase system RAM
- **Missing visualizations**: Install optional dependencies (wordcloud, scikit-learn)

## 🙏 Acknowledgments

- **Streamlit** - For the amazing web app framework
- **Plotly** - For interactive visualization capabilities
- **TextBlob** - For sentiment analysis functionality
- **Twitter** - For providing comprehensive analytics data

## 🔮 Roadmap

### **Upcoming Features**
- [ ] **Real-time API integration** for live data updates
- [ ] **Competitor analysis** dashboard
- [ ] **Automated report generation** and scheduling
- [ ] **Advanced ML models** for engagement prediction
- [ ] **Multi-account support** for agencies
- [ ] **Export to PowerPoint/PDF** functionality

### **Version History**
- **v1.0.0** - Initial release with core analytics features
- **v1.1.0** - Added word cloud and sentiment analysis
- **v1.2.0** - Enhanced viral potential analysis
- **v1.3.0** - Improved visualizations and normalized comparisons

---

**🚀 Ready to unlock insights from your Twitter data? [Get started now!](#-installation)**

⭐ **Star this repository if you find it useful!**
```

This is the complete README.md file that you can copy and save to your project repository! 🚀
