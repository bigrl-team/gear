#include <arpa/inet.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include <infinity/core/Configuration.h>
#include <infinity/utils/Address.h>
#include <infinity/utils/Debug.h>

#include "debug.h"
#include "rdma/queue_pair_factory.h"

namespace gear::rdma {
CustomQueuePairFactory::CustomQueuePairFactory(CustomContext *ctx)
    : QueuePairFactory(dynamic_cast<infinity::core::Context *>(ctx)),
      custom_context(ctx) {}

typedef struct {

  uint16_t localDeviceId;
  uint32_t queuePairNumber;
  uint32_t sequenceNumber;
  uint32_t userDataSize;
  char userData[infinity::core::Configuration::MAX_CONNECTION_USER_DATA_SIZE];

} serializedQueuePair;

CustomQueuePair *
CustomQueuePairFactory::connectToRemoteHost(const char *hostAddress,
                                            uint16_t port, void *userData,
                                            uint32_t userDataSizeInBytes) {
  INFINITY_ASSERT(
      userDataSizeInBytes <
          infinity::core::Configuration::MAX_CONNECTION_USER_DATA_SIZE,
      "[INFINITY][QUEUES][FACTORY] User data size is too large.\n")

  serializedQueuePair *receiveBuffer =
      (serializedQueuePair *)calloc(1, sizeof(serializedQueuePair));
  serializedQueuePair *sendBuffer =
      (serializedQueuePair *)calloc(1, sizeof(serializedQueuePair));

  sockaddr_in remoteAddress;
  memset(&(remoteAddress), 0, sizeof(sockaddr_in));
  remoteAddress.sin_family = AF_INET;
  inet_pton(AF_INET, hostAddress, &(remoteAddress.sin_addr));
  remoteAddress.sin_port = htons(port);

  int connectionSocket = socket(AF_INET, SOCK_STREAM, 0);
  INFINITY_ASSERT(
      connectionSocket >= 0,
      "[INFINITY][QUEUES][FACTORY] Cannot open connection socket.\n");

#ifdef GEAR_VERBOSE_DEBUG_ON
  char buffer[INET_ADDRSTRLEN];
  inet_ntop(AF_INET, &remoteAddress.sin_addr, buffer, sizeof(buffer));
  fprintf(stdout,
          "<CustomQueuePairFactory::connectToRemoteHost()> create socket %s, "
          "result %d.\n",
          buffer, connectionSocket);
#endif

  int returnValue = connect(connectionSocket, (sockaddr *)&(remoteAddress),
                            sizeof(sockaddr_in));
  INFINITY_ASSERT(returnValue == 0,
                  "[INFINITY][QUEUES][FACTORY] Could not connect to server.\n");

#ifdef GEAR_VERBOSE_DEBUG_ON
  fprintf(stdout,
          "<CustomQueuePairFactory::connectToRemoteHost()> socket connection "
          "status %d.\n",
          returnValue);
#endif

  CustomQueuePair *queuePair = new CustomQueuePair(this->custom_context);
  sendBuffer->localDeviceId = queuePair->getLocalDeviceId();
  sendBuffer->queuePairNumber = queuePair->getQueuePairNumber();
  sendBuffer->sequenceNumber = queuePair->getSequenceNumber();
  sendBuffer->userDataSize = userDataSizeInBytes;
  memcpy(sendBuffer->userData, userData, userDataSizeInBytes);
  returnValue =
      send(connectionSocket, sendBuffer, sizeof(serializedQueuePair), 0);
  INFINITY_ASSERT(returnValue == sizeof(serializedQueuePair),
                  "[INFINITY][QUEUES][FACTORY] Incorrect number of bytes "
                  "transmitted. Expected %lu. Received %d.\n",
                  sizeof(serializedQueuePair), returnValue);

  returnValue =
      recv(connectionSocket, receiveBuffer, sizeof(serializedQueuePair), 0);
  INFINITY_ASSERT(returnValue == sizeof(serializedQueuePair),
                  "[INFINITY][QUEUES][FACTORY] Incorrect number of bytes "
                  "received. Expected %lu. Received %d.\n",
                  sizeof(serializedQueuePair), returnValue);

  INFINITY_DEBUG(
      "[INFINITY][QUEUES][FACTORY] Pairing (%u, %u, %u, %u)-(%u, %u, %u, %u)\n",
      queuePair->getLocalDeviceId(), queuePair->getQueuePairNumber(),
      queuePair->getSequenceNumber(), userDataSizeInBytes,
      receiveBuffer->localDeviceId, receiveBuffer->queuePairNumber,
      receiveBuffer->sequenceNumber, receiveBuffer->userDataSize);

  queuePair->activate(receiveBuffer->localDeviceId,
                      receiveBuffer->queuePairNumber,
                      receiveBuffer->sequenceNumber);
  queuePair->setRemoteUserData(receiveBuffer->userData,
                               receiveBuffer->userDataSize);

  this->custom_context->registerQueuePair(queuePair);

  close(connectionSocket);
  free(receiveBuffer);
  free(sendBuffer);

  return queuePair;
}
} // namespace gear::rdma