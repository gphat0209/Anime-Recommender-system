# Anime Recommender System

## Introduction
The **Anime Recommender System** aims to utilize LLM's capabilities of understanding human speech to quickly rate an anime based on its description.

## Context 
Started from my personal hobby of watching anime seasonally, I always wish to have the ability to tell if the anime is good or not from its synopsis. Therefore, I aim to develop a Recommender that takes in the sypnosis and genres of an anime, then it tell me whether the anime is 'Top-tier', 'Worthwhile', 'Watchable' or 'Terrible'.

## Dataset
- The data is crawled manually on website MyAnimeList which provide useful information about various animes in the course of history. The data is then transformed to a list of objects in JSON format and fed to the LLM for training via prompt-output structure.

## Model and Algorithm
- The model I selected is a lightweight deBerta model.
- The deBerta model is then fine-tuned on the training dataset, valid dataset.
- Then I build a web interface for this project in order to deploy this as an application by FastAPI.



