FROM gcc:14
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y wget ca-certificates

# Download cpp-httplib
RUN wget https://raw.githubusercontent.com/yhirose/cpp-httplib/v0.14.3/httplib.h -O httplib.h

# Copy source files
COPY main.cpp .
COPY static ./static

# Compile with verbose output to catch errors
RUN g++ main.cpp -O2 -std=c++17 -o app -pthread -DCPPHTTPLIB_OPENSSL_SUPPORT=0 || \
    (echo "Compilation failed!" && exit 1)

# Verify the binary exists
RUN ls -lh app

EXPOSE 8080

CMD ["./app"]
