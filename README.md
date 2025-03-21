# Spam or Ham
This full-stack web application will allow a user to input the subject of their message and the message.
The application will return a likelihood of the message being spam (fake) or ham (legitimate).

## Prerequisites

- [Node.js](https://nodejs.org/) (v14 or later) and npm (v6 or later)

## Project Structure
```
cs310-final-project/ 
│
├── .ipynb_checkpoints/  
│
├── ai-sagemaker/  
│ 
├── react-front-end/ 
│ 
│ ├── public/  
│ 
│ ├── src/  
│ 
│ └── ... 
│ 
├── Dockerfile
│  
├── README.md 
```

## Setup Instructions
### 1. Clone the Repository

Open your terminal and run:

```
git clone https://github.com/<your-username>/cs310-final-project.git

cd cs310-final-project
```

### 2. Set Up the React Client
Navigate to the React client directory:
```
cd react-front-end
```

Install the necessary dependencies:
```
npm install
```

Run the application locally:
```
npm start
```