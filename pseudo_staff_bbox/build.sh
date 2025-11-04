export TEST_ENV=false
docker build \
  -t humansignal/ml-backend:v0  \
  --build-arg TEST_ENV=${TEST_ENV} \
  .