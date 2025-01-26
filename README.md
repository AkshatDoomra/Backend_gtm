1. Pretrained models, domain_knowledge and requiremnet.txt are there in the code
2. Use command ( build --no-cache -t flask-inference-service ) to build image of docker
3. Use Command ( docker run -p 5000:5000 flask-inference-service ) to run the docker on localhost 5000
4. Send post request with text snippet as a json body through postman and check retured json in response
5. csv files used for model training are also there in the repository

