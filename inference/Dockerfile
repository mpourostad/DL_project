FROM python:3.8

ENV PORT 5005
EXPOSE $PORT
# Set up a working folder and install the pre-reqs
WORKDIR /app
ADD ./requirements.txt /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
ADD ./res18-unet.pt /app/res18-unet.pt
ADD ./models.py /app/models.py
ADD ./test_set /app/test_set
ADD ./loss.py /app/loss.py
ADD ./utils.py /app/utils.py
ADD ./200ep_resnet-18Final_gan.pt /app/200ep_resnet-18Final_gan.pt
ADD ./templates /app/templates
ADD ./results /app/results
ADD ./main.py  /app/main.py
# Run the service
CMD [ "python", "main.py"]