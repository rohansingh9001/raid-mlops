# raid-mlops
Repository for a live demo for RAID MLOps session 2023

The repository contains source code for a web application that can classify News Headlines into 20 different classes.

The dataset used for this project was the 20 Newsgroups dataset from [scikit-learn](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html).

# Instructions

Clone the git repository into your local system `git clone https://github.com/rohansingh9001/raid-mlops` there are multiple ways to set up the project.

## Without Docker and Docker Compose.

1) Create a new python 3.10 environment for this project using conda (or any other alternatives) `conda create -n mlops python=3.10`.

2) Install python dependencies using pip `python -m pip install -r requirements.txt --no-cache-dir`

3) Run the backend server `python app.py`

> Note: This README assumes that the pre-trained models are already present on your local system in directories named `models` and `tokenizer` respectively. If the models are not present, or you wish to train your own models, before running the backend server, run the `train.py` file by running `python train.py`. This should store a pre-trained model on your disk.

4) The backend server is running. You should see uvicorn running the FastAPI app. Run `curl localhost:8000/ping` to test for a response. If the response is `pong` your server is live.

5) Optional: Start the frontend server using react. You should have node (preferably v16) on your system.

6) Change the IP address of the backend: Open `client/src/components/QueryForm.jsx` and change the API call to use your localhost.
FROM
   ```
   const response = await fetch('http://54.169.248.184:443/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });
   ```
TO
   ```
   const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });
  ```
7) Install dependencies - `cd client` and `npm install`
8) Run the development server using `npm start`
9) Visit `localhost:3000` on your browser to see the frontend application.

## Setup with Docker and Docker Compose
1) Follow step 6 given above.
2) Run `docker-compose up -d --build` The backend server should be running on port 8000. The frontend server is hosted on port 80.
3) Visit `localhost` or `127.0.0.1` on your browser to visit the application.
