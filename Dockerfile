FROM gcc:14
WORKDIR /app
COPY . .
RUN apt-get update && apt-get install -y wget && \
    wget https://raw.githubusercontent.com/yhirose/cpp-httplib/master/httplib.h
RUN g++ main.cpp -O2 -std=c++17 -o app -pthread
CMD ["./app"]
