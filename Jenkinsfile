pipeline {
  agent any

  stages {
    stage('Build') {
      steps {
        sh 'python -m venv venv'
        sh 'venv/bin/pip install -r requirements.txt'
      }
    }
    stage('Test') {
      steps {
        sh 'venv/bin/python unit_tests.py'
      }
    }
  }
}
