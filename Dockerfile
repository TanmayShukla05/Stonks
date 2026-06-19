FROM gcc:14

WORKDIR /app

COPY . .

RUN g++ main.cpp -O2 -std=c++17 -o app

CMD ["./app"]