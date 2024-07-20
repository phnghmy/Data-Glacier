from flask import Flask, request, jsonify, send_file, render_template
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

app = Flask(__name__)

# Load the model
model = joblib.load('kmeans_model.pkl')

# Load the data with clusters
df = pd.read_csv('songs_with_clusters.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    popularity = np.array(data['popularity']).reshape(-1, 1)
    prediction = model.predict(popularity)
    return jsonify({'cluster': prediction.tolist()})

@app.route('/plot')
def plot():
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Popularity'], np.zeros_like(df['Popularity']), c=df['Cluster'], cmap='viridis')
    plt.scatter(model.cluster_centers_, np.zeros_like(model.cluster_centers_), s=300, c='red', label='Centroids')
    plt.title('K-Means Clustering of Popularity')
    plt.xlabel('Popularity')
    plt.yticks([])  # Hide y-axis ticks
    plt.legend()
    plot_path = "static/plot.png"
    plt.savefig(plot_path)
    plt.close()
    return "Plot created successfully."

@app.route('/create_pdf')
def create_pdf():
    pdf_path = "static/Deployment_Document.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    c.drawString(100, height - 50, "Name: Ha My")
    c.drawString(100, height - 70, "Batch Code: LISUM34")
    c.drawString(100, height - 90, "Submission Date: July17")
    c.drawString(100, height - 110, "Submitted to: DataGlacier")

    c.drawString(100, height - 150, "Step 1: Load Data")
    c.drawString(100, height - 170, "Spotify songs dataset")

    c.drawString(100, height - 200, "Step 2: Train and Save Model")
    c.drawString(100, height - 220, "Popularity kmeans classifications")

    c.drawString(100, height - 250, "Step 3: Flask Deployment")
    c.drawString(100, height - 270, "Web Deploy")

    c.save()
    return "PDF created successfully."

@app.route('/download_pdf')
def download_pdf():
    return send_file('static/Deployment_Document.pdf', as_attachment=True)

@app.route('/view_pdf')
def view_pdf():
    return render_template('view_pdf.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5005)