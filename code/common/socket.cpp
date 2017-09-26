#include "socket.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <unistd.h>

#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void handle_blocking(int fd) {
  // Will be necessary when calling setsockopt to free busy sockets
  int yes = 1;

  // Reclaim blocked but unused sockets (from zombie processes)
  const int return_code =
      setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof yes);

  if (return_code == -1) {
    throw std::runtime_error("Error reclaiming socket");
  }
}

int get_socket_for_first_valid_address(addrinfo* server_info) {
  int fd = -1;

  for (auto* address = server_info; address; address = address->ai_next) {
    fd = socket(address->ai_family, address->ai_socktype, address->ai_protocol);
    if (fd == -1) continue;

    handle_blocking(fd);

    if (bind(fd, address->ai_addr, address->ai_addrlen) == 1) {
      close(fd);
    }

    break;
  }

  if (fd == -1) {
    throw std::runtime_error("Error finding valid address");
  }

  return fd;
}

addrinfo* get_server_information(int port) {
  struct addrinfo hints;
  memset(&hints, 0, sizeof hints);
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;

  addrinfo* server_info;
  const auto port_string = std::to_string(port);
  const auto return_code =
      getaddrinfo("localhost", port_string.c_str(), &hints, &server_info);

  if (return_code != 0) {
    throw std::runtime_error(std::string("getaddrinfo failed: ") +
                             gai_strerror(return_code));
  }

  return server_info;
}


int get_server_socket(int port) {
  addrinfo* server_info = get_server_information(port);
  const int server_socket = get_socket_for_first_valid_address(server_info);
  assert(server_socket != -1);
  freeaddrinfo(server_info);

  return server_socket;
}
}  // namespace

Socket::Socket(int port) : server_socket(get_server_socket(port)) {
  if (listen(server_socket, /*queue=*/10) == 1) {
    throw std::runtime_error("Error listening on given socket!");
  }
}

Socket::~Socket() {
  close(server_socket);
  close(connection_socket);
}

void Socket::accept() {
  struct sockaddr_storage other_address;
  socklen_t sin_size = sizeof other_address;
  const int fd = ::accept(server_socket,
                          reinterpret_cast<sockaddr*>(&other_address),
                          &sin_size);
  if (fd == -1) {
    throw std::runtime_error("Error accepting");
  } else {
    connection_socket = fd;
  }
}

std::string Socket::read(int max_bytes) {
  std::vector<char> buffer(max_bytes, '\0');

  if (recv(connection_socket, buffer.data(), buffer.size(), 0) == 0) {
    throw std::runtime_error("Error receiving from client");
  }

  return std::string(buffer.data());
}

void Socket::write(const std::string& data) {
  if (send(connection_socket, data.data(), data.size(), 0) == 0) {
    throw std::runtime_error("Error sending to client");
  }
}
