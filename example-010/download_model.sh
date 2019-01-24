mkdir api_project
cd api_project
git init
git remote add -f origin https://github.com/tensorflow/models.git
git config core.sparseCheckout true
echo "research/object_detection/*" >> .git/info/sparse-checkout
git pull origin master
