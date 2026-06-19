FROM gcc:14
WORKDIR /app

RUN apt-get update && apt-get install -y wget ca-certificates
RUN wget https://raw.githubusercontent.com/yhirose/cpp-httplib/v0.14.3/httplib.h -O httplib.h

COPY main.cpp .
COPY static ./static

RUN g++ main.cpp -O2 -std=c++17 -o app -pthread

CMD ["./app"]
