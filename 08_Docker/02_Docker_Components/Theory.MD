# **Docker Engine**
Docker Engine is the core component of the Docker Platform, responsible for creating, running and managing Docker containers, it serves as the runtime that powers Docker's containerization capabilities. 

[Image of the docker]
Docker Daemon is the main thing. 
First we give some commands to the CLI and then CLI tells to the REST API and then that API tells to the Daemon for working... So, Docker Daemon or server is the most important thing in this above illustration.

We can send: images, containers, network and data volumes from the CLI to the docker daemon.

## **Components of Docker Engine**
- **Docker Daemon(dockerd):**
    - **Function:** The docker daemon is the background service running on the host machine. It manages docker objects such as: images, containers, networks and volumes.
    - **Interaction:** It listens for Docker API requests and processes them, handling container lifecycle operations (start, stop, restart, etc).
- **Docker CLI(docker):**
    - **Function:** The docker command line interface(CLI) is the tool that users interact with to communicate with the Docker daemon
    - **Usage:** Users run docker commands through the CLI to perform tasks like building images, running containers, and managing Docker resources.
- **REST API:**
    - **Function:** The docker REST API allows communication between the Docker CLI and the docker daemon. It also enables programmatic interaction with Docker.
    - **Usage:** Developers can use the API to automate Docker operations or integrate Docker functionality into their applications.

## **Docker Image**
A docker image is a lightweight, stand-alone, and executable software package that includes everything needed to run a piece of software, such as the code, runtime, libraries, environment variables, and configuration files. Images are used to create Docker containers which are instances of these images.

**Components of a Docker Image**
- **Base Image:** The starting point for building an image. It could be a minimal OS image like alpine, a full-fledged OS like: Ubuntu or even another application image like: python or node.
- **Application Code:** The actual code and files necessary for the application to run.
- **Dependencies:** Libraries, Frameworks and packages required by the application.
- **Metadata:** Information about the image, such as environment variables, labels and exposed ports.

**Docker Image LifeCycle**
- **Creation:** Images are created using docker build command, which processes the instructions in a Dockerfile to create the image layers.
- **Storage:** Images are stored locally on the host machine. They can also be pushed to and pulled from Docker registries like: DockerHub, AWS ECR, or Google Container Registry.
- **Distribution:** Images can be shared by pushing them to a Docker registry, allowing others to pull and use the same image.
- **Execution:** Images are executed by running containers, which are instances of these images

## **DockerFile**
A dockerfile is a text file that contains a series of instructions used to build a docker image. Each instruction in a dockerfile creates a layer in the image, allowing for efficient image creation and reuse of layers. Dockerfiles are used to automate the image creation process, ensuring consistency and reproducibility.

**Key Components of a DockerFile**
- **Base Image (FROM):** Specifies the starting point for the image, which could be a minimal operating system, a specific version of a language runtime or another image.
- **Labels (LABEL):** Adds metadata to the image, such as version, description, or maintainer.
- **Run Commands (RUN):** Executes commands in the image during the build process, typically used to install software packages.
- **Copy Files (COPY):** Copies files or directories from the host system to the image
- **Environment Variables (ENV):** Sets env variables in the iamge
- **Work Directory (WORKDIR):** Sets the working directory for subsequent instructions.
- **Expose Ports (EXPOSE):** Informs Docker that the container listens on specified network.
- **Command (CMD):** Provides a default command to run when the container starts
- **Volume (VOLUME):** Creates a mount point with a specified path and marks it as holding externally mounted volumes from the host or other containers
- **Arguments (ARG):** Defines build-time variables.

```python
# Use an official python runtime as a base image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY ./app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]
```

## **Docker Container** 
A docker container is a lightweight, portable, and isolated environment that encapsulated an application and its dependencies, allowing it to run consistently across different computing environments. Containers are created from docker images, which are immutable and contain all the necessary components for the application to run.

***Summary:*** When we execute the docker image and the instance we get that is called the docker container.
***Example:*** DVD is the image, when we use to play on TV it is the instance that is playing so that will be the container. 

## **Registry**
A docker registry is a service that stores and distributes Docker images. It acts as a repository where users can push, pull, and manages Docker images. Docker Hub is the most well-known publish registry, but private registries can also be set up to securely store and manage images within an organization.

In the docker image life cycle -> the storage part is called the registry. 

**Key Components of a Docker Registry**
- **Repositories:** A repository is a collection of related Docker images, typically different versions of the same application. Each repository can hold multiple tags, representing different versions of an image.
- **Tags:** Tags are used to version images within a repository. For example: myapp:1.0, myapp:2.0 and myapp:latest are tags for different versions of the myapp image.

**Types of Docker Registries**
- **Docker Hub:**
    - **Description:** The default public registry provided by docker, which hosts a vast number of public images and also supports private repositories.
    - **URL:** hub.docker.com
    - **Use Case:** Publicly sharing images and accessing a large collection of pre-built images from the community and official repositories.
- **Private Registries:**
    - **Description:** Custom registries set up by organizations to securely store and manage their own Docker images
    - **Use case:** Ensuring security and control over image distribution within an organization.
- **Third-party registries:**
    - **Examples:** Amazon Elastic Container Registry (ECR), Google Container Registry (GCR), Azure Container Registry (ACR).
    - **Use Case:** Integrating with cloud platforms for seamless deployment and management of images within cloud infrastructure.

**Benefits of Using Docker Registries**
- **Centralised Image Management:** Registries provide a centralized location to store and manage Docker images, making it easier to organize and distribute them.
- **Version Control:** Using tags, registries allow version control of images, enabling users to easily roll back to prev versions if needed.
- **Collaboration:** Public registries like: Docker Hub faciliate collaboration by allowing users to share images with the community or within teams.
- **Security:** Private registries ensure that sensitive images are stored securely and access is controlled within an organization.
- **Integration with CI/CD:** Registries integrate seamlessly with CI/CD pipelines, automating the process of building, storing, and deploying Docker images.

