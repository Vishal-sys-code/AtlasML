## **Workflow of Pulling image from the Registry**
Suppose, I have uploaded the project to the dockerhub. And to pull that image in the docker desktop we need to run the command:
```
docker pull username/image_name
```

<hr>

To run the image in the docker desktop:
```
docker run -p 8501:8501 username/image_name
```

This concept is called port mapping

## **Workflow to Dockerise our project from our local to the Registry**
- Run your project
- Make a file called: `Dockerfile`
- In that file you need to put all the instructions that how to run your project

**Example:**
```
# Base Image
FROM python 3.x

# WorkDIR
WORKDIR /app

# Copy
COPY . /app

# Run
RUN pip install -r requirements.txt

# Ports
EXPOSE 5000 OR 8000 OR 8001 OR 8501

# Commands
CMD ["python", "./app.py"]
```

To build the image you need to run a command that is:
```
docker build -t username/name_of_project .
```

Go to the docker desktop -> Select Images -> You can see that you have build something.
<br>
Now to run in our desktop do the port mapping
```
docker run -p port_you_wanna_give:port_used_in_docker username/image_name
```

Example:
```
docker run -p 8888:5000 username/iamge_name
```
