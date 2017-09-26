#ifndef SOCKET_H
#define SOCKET_H

#include <string>

class Socket {
 public:
  explicit Socket(int port);

  ~Socket();

  void accept();

  std::string read(int max_bytes);

  void write(const std::string& data);

 private:
  const int server_socket{-1};
  int connection_socket{-1};
};

#endif  // SOCKET_H
