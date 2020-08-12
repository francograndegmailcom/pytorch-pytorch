  pytorch_doc_push:
    resource_class: medium
    machine:
      image: ubuntu-1604:201903-01
    steps:
    - attach_workspace:
        at: /tmp/workspace
    - run:
        name: Docs push
        command: |
          pushd /tmp/workspace
          /usr/bin/expect \<<DONE
            spawn git push -u origin master
            expect "Username*"
            send "pytorchbot\n"
            expect "Password*"
            send "$::env(GITHUB_PYTORCHBOT_TOKEN)\n"
            expect eof
          DONE

  pytorch_python_doc_build:
    environment:
      BUILD_ENVIRONMENT: pytorch-python-doc-push
      # TODO: stop hardcoding this
      DOCKER_IMAGE: "308535385114.dkr.ecr.us-east-1.amazonaws.com/pytorch/pytorch-linux-xenial-py3.6-gcc5.4:8bdba785b1eac4d297d5f5930f979518012a56e0"
    resource_class: large
    machine:
      image: ubuntu-1604:201903-01
    steps:
    - checkout
    - setup_linux_system_environment
    - setup_ci_environment
    - run:
        name: Doc Build and Push
        no_output_timeout: "1h"
        command: |
          set -ex
          export COMMIT_DOCKER_IMAGE=${DOCKER_IMAGE}-${CIRCLE_SHA1}
          echo "DOCKER_IMAGE: "${COMMIT_DOCKER_IMAGE}
          time docker pull ${COMMIT_DOCKER_IMAGE} >/dev/null
          export id=$(docker run --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -t -d -w /var/lib/jenkins ${COMMIT_DOCKER_IMAGE})

          export COMMAND='((echo "export BUILD_ENVIRONMENT=${BUILD_ENVIRONMENT}" && echo "source ./workspace/env" && echo "sudo chown -R jenkins workspace && cd workspace && . ./.circleci/scripts/python_doc_push_script.sh docs/master master site") | docker exec -u jenkins -i "$id" bash) 2>&1'

          echo ${COMMAND} > ./command.sh && unbuffer bash ./command.sh | ts

          mkdir -p ~/workspace/build_artifacts
          docker cp $id:/var/lib/jenkins/workspace/pytorch.github.io/docs/master ~/workspace/build_artifacts
          docker cp $id:/var/lib/jenkins/workspace/pytorch.github.io /tmp/workspace

          # Save the docs build so we can debug any problems
          export DEBUG_COMMIT_DOCKER_IMAGE=${COMMIT_DOCKER_IMAGE}-debug
          docker commit "$id" ${DEBUG_COMMIT_DOCKER_IMAGE}
          time docker push ${DEBUG_COMMIT_DOCKER_IMAGE}
    - persist_to_workspace:
        root: /tmp/workspace
        paths:
          - .
    - store_artifacts:
        path: ~/workspace/build_artifacts/master
        destination: docs

  pytorch_cpp_doc_build:
    environment:
      BUILD_ENVIRONMENT: pytorch-cpp-doc-push
      DOCKER_IMAGE: "308535385114.dkr.ecr.us-east-1.amazonaws.com/pytorch/pytorch-linux-xenial-py3.6-gcc5.4:8bdba785b1eac4d297d5f5930f979518012a56e0"
    resource_class: large
    machine:
      image: ubuntu-1604:201903-01
    steps:
    - checkout
    - setup_linux_system_environment
    - setup_ci_environment
    - run:
        name: Doc Build and Push
        no_output_timeout: "1h"
        command: |
          set -ex
          export COMMIT_DOCKER_IMAGE=${DOCKER_IMAGE}-${CIRCLE_SHA1}
          echo "DOCKER_IMAGE: "${COMMIT_DOCKER_IMAGE}
          time docker pull ${COMMIT_DOCKER_IMAGE} >/dev/null
          export id=$(docker run --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -t -d -w /var/lib/jenkins ${COMMIT_DOCKER_IMAGE})

          export COMMAND='((echo "export BUILD_ENVIRONMENT=${BUILD_ENVIRONMENT}" && echo "source ./workspace/env" && echo "sudo chown -R jenkins workspace && cd workspace && . ./.circleci/scripts/cpp_doc_push_script.sh docs/master master") | docker exec -u jenkins -i "$id" bash) 2>&1'

          echo ${COMMAND} > ./command.sh && unbuffer bash ./command.sh | ts

          mkdir -p ~/workspace/build_artifacts
          docker cp $id:/var/lib/jenkins/workspace/cppdocs/ /tmp/workspace

          # Save the docs build so we can debug any problems
          export DEBUG_COMMIT_DOCKER_IMAGE=${COMMIT_DOCKER_IMAGE}-debug
          docker commit "$id" ${DEBUG_COMMIT_DOCKER_IMAGE}
          time docker push ${DEBUG_COMMIT_DOCKER_IMAGE}

    - persist_to_workspace:
        root: /tmp/workspace
        paths:
          - .

  pytorch_macos_10_13_py3_build:
    environment:
      BUILD_ENVIRONMENT: pytorch-macos-10.13-py3-build
    macos:
      xcode: "9.4.1"
    steps:
      - checkout
      - run_brew_for_macos_build
      - run:
          name: Build
          no_output_timeout: "1h"
          command: |
            set -e
            export IN_CIRCLECI=1

            # Install sccache
            sudo curl --retry 3 https://s3.amazonaws.com/ossci-macos/sccache --output /usr/local/bin/sccache
            sudo chmod +x /usr/local/bin/sccache
            export SCCACHE_BUCKET=ossci-compiler-cache-circleci-v2

            # This IAM user allows write access to S3 bucket for sccache
            set +x
            export AWS_ACCESS_KEY_ID=${CIRCLECI_AWS_ACCESS_KEY_FOR_SCCACHE_S3_BUCKET_V4}
            export AWS_SECRET_ACCESS_KEY=${CIRCLECI_AWS_SECRET_KEY_FOR_SCCACHE_S3_BUCKET_V4}
            set -x

            chmod a+x .jenkins/pytorch/macos-build.sh
            unbuffer .jenkins/pytorch/macos-build.sh 2>&1 | ts

      - persist_to_workspace:
          root: /Users/distiller/workspace/
          paths:
            - miniconda3

  pytorch_macos_10_13_py3_test:
    environment:
      BUILD_ENVIRONMENT: pytorch-macos-10.13-py3-test
    macos:
      xcode: "9.4.1"
    steps:
      - checkout
      - attach_workspace:
          at: ~/workspace
      - run_brew_for_macos_build
      - run:
          name: Test
          no_output_timeout: "1h"
          command: |
            set -e
            export IN_CIRCLECI=1

            chmod a+x .jenkins/pytorch/macos-test.sh
            unbuffer .jenkins/pytorch/macos-test.sh 2>&1 | ts
      - store_test_results:
          path: test/test-reports

  pytorch_android_gradle_build:
    environment:
      BUILD_ENVIRONMENT: pytorch-linux-xenial-py3-clang5-android-ndk-r19c-gradle-build
      DOCKER_IMAGE: "308535385114.dkr.ecr.us-east-1.amazonaws.com/pytorch/pytorch-linux-xenial-py3-clang5-android-ndk-r19c:8bdba785b1eac4d297d5f5930f979518012a56e0"
      PYTHON_VERSION: "3.6"
    resource_class: large
    machine:
      image: ubuntu-1604:201903-01
    steps:
    - checkout
    - setup_linux_system_environment
    - setup_ci_environment
    - run:
        name: pytorch android gradle build
        no_output_timeout: "1h"
        command: |
          set -eux
          docker_image_commit=${DOCKER_IMAGE}-${CIRCLE_SHA1}

          docker_image_libtorch_android_x86_32=${docker_image_commit}-android-x86_32
          docker_image_libtorch_android_x86_64=${docker_image_commit}-android-x86_64
          docker_image_libtorch_android_arm_v7a=${docker_image_commit}-android-arm-v7a
          docker_image_libtorch_android_arm_v8a=${docker_image_commit}-android-arm-v8a

          echo "docker_image_commit: "${docker_image_commit}
          echo "docker_image_libtorch_android_x86_32: "${docker_image_libtorch_android_x86_32}
          echo "docker_image_libtorch_android_x86_64: "${docker_image_libtorch_android_x86_64}
          echo "docker_image_libtorch_android_arm_v7a: "${docker_image_libtorch_android_arm_v7a}
          echo "docker_image_libtorch_android_arm_v8a: "${docker_image_libtorch_android_arm_v8a}

          # x86_32
          time docker pull ${docker_image_libtorch_android_x86_32} >/dev/null
          export id_x86_32=$(docker run --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -t -d -w /var/lib/jenkins ${docker_image_libtorch_android_x86_32})

          export COMMAND='((echo "source ./workspace/env" && echo "sudo chown -R jenkins workspace") | docker exec -u jenkins -i "$id_x86_32" bash) 2>&1'
          echo ${COMMAND} > ./command.sh && unbuffer bash ./command.sh | ts

          # arm-v7a
          time docker pull ${docker_image_libtorch_android_arm_v7a} >/dev/null
          export id_arm_v7a=$(docker run --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -t -d -w /var/lib/jenkins ${docker_image_libtorch_android_arm_v7a})

          export COMMAND='((echo "source ./workspace/env" && echo "sudo chown -R jenkins workspace") | docker exec -u jenkins -i "$id_arm_v7a" bash) 2>&1'
          echo ${COMMAND} > ./command.sh && unbuffer bash ./command.sh | ts

          mkdir -p ~/workspace/build_android_install_arm_v7a
          docker cp $id_arm_v7a:/var/lib/jenkins/workspace/build_android/install ~/workspace/build_android_install_arm_v7a

          # x86_64
          time docker pull ${docker_image_libtorch_android_x86_64} >/dev/null
          export id_x86_64=$(docker run --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -t -d -w /var/lib/jenkins ${docker_image_libtorch_android_x86_64})

          export COMMAND='((echo "source ./workspace/env" && echo "sudo chown -R jenkins workspace") | docker exec -u jenkins -i "$id_x86_64" bash) 2>&1'
          echo ${COMMAND} > ./command.sh && unbuffer bash ./command.sh | ts

          mkdir -p ~/workspace/build_android_install_x86_64
          docker cp $id_x86_64:/var/lib/jenkins/workspace/build_android/install ~/workspace/build_android_install_x86_64

          # arm-v8a
          time docker pull ${docker_image_libtorch_android_arm_v8a} >/dev/null
          export id_arm_v8a=$(docker run --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -t -d -w /var/lib/jenkins ${docker_image_libtorch_android_arm_v8a})

          export COMMAND='((echo "source ./workspace/env" && echo "sudo chown -R jenkins workspace") | docker exec -u jenkins -i "$id_arm_v8a" bash) 2>&1'
          echo ${COMMAND} > ./command.sh && unbuffer bash ./command.sh | ts

          mkdir -p ~/workspace/build_android_install_arm_v8a
          docker cp $id_arm_v8a:/var/lib/jenkins/workspace/build_android/install ~/workspace/build_android_install_arm_v8a

          docker cp ~/workspace/build_android_install_arm_v7a $id_x86_32:/var/lib/jenkins/workspace/build_android_install_arm_v7a
          docker cp ~/workspace/build_android_install_x86_64 $id_x86_32:/var/lib/jenkins/workspace/build_android_install_x86_64
          docker cp ~/workspace/build_android_install_arm_v8a $id_x86_32:/var/lib/jenkins/workspace/build_android_install_arm_v8a

          # run gradle buildRelease
          export COMMAND='((echo "source ./workspace/env" && echo "export BUILD_ENVIRONMENT=${BUILD_ENVIRONMENT}" && echo "export GRADLE_OFFLINE=1" && echo "sudo chown -R jenkins workspace && cd workspace && ./.circleci/scripts/build_android_gradle.sh") | docker exec -u jenkins -i "$id_x86_32" bash) 2>&1'
          echo ${COMMAND} > ./command.sh && unbuffer bash ./command.sh | ts

          mkdir -p ~/workspace/build_android_artifacts
          docker cp $id_x86_32:/var/lib/jenkins/workspace/android/artifacts.tgz ~/workspace/build_android_artifacts/

          output_image=$docker_image_libtorch_android_x86_32-gradle
          docker commit "$id_x86_32" ${output_image}
          time docker push ${output_image}
    - upload_binary_size_for_android_build:
        build_type: prebuilt
        artifacts: /home/circleci/workspace/build_android_artifacts/artifacts.tgz
    - store_artifacts:
        path: ~/workspace/build_android_artifacts/artifacts.tgz
        destination: artifacts.tgz

  pytorch_android_publish_snapshot:
    environment:
      BUILD_ENVIRONMENT: pytorch-linux-xenial-py3-clang5-android-ndk-r19c-gradle-publish-snapshot
      DOCKER_IMAGE: "308535385114.dkr.ecr.us-east-1.amazonaws.com/pytorch/pytorch-linux-xenial-py3-clang5-android-ndk-r19c:8bdba785b1eac4d297d5f5930f979518012a56e0"
      PYTHON_VERSION: "3.6"
    resource_class: large
    machine:
      image: ubuntu-1604:201903-01
    steps:
    - checkout
    - setup_linux_system_environment
    - checkout
    - setup_ci_environment
    - run:
        name: pytorch android gradle build
        no_output_timeout: "1h"
        command: |
          set -eux
          docker_image_commit=${DOCKER_IMAGE}-${CIRCLE_SHA1}

          docker_image_libtorch_android_x86_32_gradle=${docker_image_commit}-android-x86_32-gradle

          echo "docker_image_commit: "${docker_image_commit}
          echo "docker_image_libtorch_android_x86_32_gradle: "${docker_image_libtorch_android_x86_32_gradle}

          # x86_32
          time docker pull ${docker_image_libtorch_android_x86_32_gradle} >/dev/null
          export id_x86_32=$(docker run --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -t -d -w /var/lib/jenkins ${docker_image_libtorch_android_x86_32_gradle})

          export COMMAND='((echo "source ./workspace/env" && echo "sudo chown -R jenkins workspace" && echo "export BUILD_ENVIRONMENT=${BUILD_ENVIRONMENT}" && echo "export SONATYPE_NEXUS_USERNAME=${SONATYPE_NEXUS_USERNAME}" && echo "export SONATYPE_NEXUS_PASSWORD=${SONATYPE_NEXUS_PASSWORD}" && echo "export ANDROID_SIGN_KEY=${ANDROID_SIGN_KEY}" && echo "export ANDROID_SIGN_PASS=${ANDROID_SIGN_PASS}" && echo "sudo chown -R jenkins workspace && cd workspace && ./.circleci/scripts/publish_android_snapshot.sh") | docker exec -u jenkins -i "$id_x86_32" bash) 2>&1'
          echo ${COMMAND} > ./command.sh && unbuffer bash ./command.sh | ts

          output_image=${docker_image_libtorch_android_x86_32_gradle}-publish-snapshot
          docker commit "$id_x86_32" ${output_image}
          time docker push ${output_image}

  pytorch_android_gradle_build-x86_32:
    environment:
      BUILD_ENVIRONMENT: pytorch-linux-xenial-py3-clang5-android-ndk-r19c-gradle-build-only-x86_32
      DOCKER_IMAGE: "308535385114.dkr.ecr.us-east-1.amazonaws.com/pytorch/pytorch-linux-xenial-py3-clang5-android-ndk-r19c:8bdba785b1eac4d297d5f5930f979518012a56e0"
      PYTHON_VERSION: "3.6"
    resource_class: large
    machine:
      image: ubuntu-1604:201903-01
    steps:
    - checkout
    - setup_linux_system_environment
    - checkout
    - setup_ci_environment
    - run:
        name: pytorch android gradle build only x86_32 (for PR)
        no_output_timeout: "1h"
        command: |
          set -e
          docker_image_libtorch_android_x86_32=${DOCKER_IMAGE}-${CIRCLE_SHA1}-android-x86_32
          echo "docker_image_libtorch_android_x86_32: "${docker_image_libtorch_android_x86_32}

          # x86
          time docker pull ${docker_image_libtorch_android_x86_32} >/dev/null
          export id=$(docker run --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -t -d -w /var/lib/jenkins ${docker_image_libtorch_android_x86_32})

          export COMMAND='((echo "source ./workspace/env" && echo "export BUILD_ENVIRONMENT=${BUILD_ENVIRONMENT}" && echo "export GRADLE_OFFLINE=1" && echo "sudo chown -R jenkins workspace && cd workspace && ./.circleci/scripts/build_android_gradle.sh") | docker exec -u jenkins -i "$id" bash) 2>&1'
          echo ${COMMAND} > ./command.sh && unbuffer bash ./command.sh | ts

          mkdir -p ~/workspace/build_android_x86_32_artifacts
          docker cp $id:/var/lib/jenkins/workspace/android/artifacts.tgz ~/workspace/build_android_x86_32_artifacts/

          output_image=${docker_image_libtorch_android_x86_32}-gradle
          docker commit "$id" ${output_image}
          time docker push ${output_image}
    - upload_binary_size_for_android_build:
        build_type: prebuilt-single
        artifacts: /home/circleci/workspace/build_android_x86_32_artifacts/artifacts.tgz
    - store_artifacts:
        path: ~/workspace/build_android_x86_32_artifacts/artifacts.tgz
        destination: artifacts.tgz

  pytorch_android_gradle_custom_build_single:
    environment:
      BUILD_ENVIRONMENT: pytorch-linux-xenial-py3-clang5-android-ndk-r19c-gradle-custom-build-single
      DOCKER_IMAGE: "308535385114.dkr.ecr.us-east-1.amazonaws.com/pytorch/pytorch-linux-xenial-py3-clang5-android-ndk-r19c:209062ef-ab58-422a-b295-36c4eed6e906"
      PYTHON_VERSION: "3.6"
    resource_class: large
    machine:
      image: ubuntu-1604:201903-01
    steps:
    - checkout
    - setup_linux_system_environment
    - checkout
    - setup_ci_environment
    - run:
        name: pytorch android gradle custom build single architecture (for PR)
        no_output_timeout: "1h"
        command: |
          set -e
          # Unlike other gradle jobs, it's not worth building libtorch in a separate CI job and share via docker, because:
          # 1) Not shareable: it's custom selective build, which is different from default libtorch mobile build;
          # 2) Not parallelizable by architecture: it only builds libtorch for one architecture;

          echo "DOCKER_IMAGE: ${DOCKER_IMAGE}"
          time docker pull ${DOCKER_IMAGE} >/dev/null
          export id=$(docker run --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -t -d -w /var/lib/jenkins ${DOCKER_IMAGE})

          git submodule sync && git submodule update -q --init --recursive

          VOLUME_MOUNTS="-v /home/circleci/project/:/var/lib/jenkins/workspace"
          export id=$(docker run ${VOLUME_MOUNTS} --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -t -d -w /var/lib/jenkins ${DOCKER_IMAGE})

          export COMMAND='((echo "source ./workspace/env" && echo "export BUILD_ENVIRONMENT=${BUILD_ENVIRONMENT}" && echo "export GRADLE_OFFLINE=1" && echo "sudo chown -R jenkins workspace && cd workspace && ./.circleci/scripts/build_android_gradle.sh") | docker exec -u jenkins -i "$id" bash) 2>&1'
          echo ${COMMAND} > ./command.sh && unbuffer bash ./command.sh | ts

          # Skip docker push as this job is purely for size analysis purpose.
          # Result binaries are already in `/home/circleci/project/` as it's mounted instead of copied.

    - upload_binary_size_for_android_build:
        build_type: custom-build-single

  pytorch_ios_build:
    <<: *pytorch_ios_params
    macos:
      xcode: "11.2.1"
    steps:
      - checkout
      - run_brew_for_ios_build
      - run:
          name: Run Fastlane
          no_output_timeout: "1h"
          command: |
            set -e
            PROJ_ROOT=/Users/distiller/project
            cd ${PROJ_ROOT}/ios/TestApp
            # install fastlane
            sudo gem install bundler && bundle install
            # install certificates
            echo ${IOS_CERT_KEY} >> cert.txt
            base64 --decode cert.txt -o Certificates.p12
            rm cert.txt
            bundle exec fastlane install_cert
            # install the provisioning profile
            PROFILE=TestApp_CI.mobileprovision
            PROVISIONING_PROFILES=~/Library/MobileDevice/Provisioning\ Profiles
            mkdir -pv "${PROVISIONING_PROFILES}"
            cd "${PROVISIONING_PROFILES}"
            echo ${IOS_SIGN_KEY} >> cert.txt
            base64 --decode cert.txt -o ${PROFILE}
            rm cert.txt
      - run:
          name: Build
          no_output_timeout: "1h"
          command: |
            set -e
            export IN_CIRCLECI=1
            WORKSPACE=/Users/distiller/workspace
            PROJ_ROOT=/Users/distiller/project
            export TCLLIBPATH="/usr/local/lib"

            # Install conda
            curl --retry 3 -o ~/conda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
            chmod +x ~/conda.sh
            /bin/bash ~/conda.sh -b -p ~/anaconda
            export PATH="~/anaconda/bin:${PATH}"
            source ~/anaconda/bin/activate

            # Install dependencies
            retry () {
                $*  || (sleep 1 && $*) || (sleep 2 && $*) || (sleep 4 && $*) || (sleep 8 && $*)
            }

            retry conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing requests --yes

            # sync submodules
            cd ${PROJ_ROOT}
            git submodule sync
            git submodule update --init --recursive

            # export
            export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

            # run build script
            chmod a+x ${PROJ_ROOT}/scripts/build_ios.sh
            echo "IOS_ARCH: ${IOS_ARCH}"
            echo "IOS_PLATFORM: ${IOS_PLATFORM}"

            #check the custom build flag
            echo "SELECTED_OP_LIST: ${SELECTED_OP_LIST}"
            if [ -n "${SELECTED_OP_LIST}" ]; then
                export SELECTED_OP_LIST="${PROJ_ROOT}/ios/TestApp/custom_build/${SELECTED_OP_LIST}"
            fi
            export IOS_ARCH=${IOS_ARCH}
            export IOS_PLATFORM=${IOS_PLATFORM}
            unbuffer ${PROJ_ROOT}/scripts/build_ios.sh 2>&1 | ts
      - run:
          name: Run Build Test
          no_output_timeout: "30m"
          command: |
            set -e
            PROJ_ROOT=/Users/distiller/project
            PROFILE=TestApp_CI
            # run the ruby build script
            if ! [ -x "$(command -v xcodebuild)" ]; then
              echo 'Error: xcodebuild is not installed.'
              exit 1
            fi
            echo ${IOS_DEV_TEAM_ID}
            if [ ${IOS_PLATFORM} != "SIMULATOR" ]; then
              ruby ${PROJ_ROOT}/scripts/xcode_build.rb -i ${PROJ_ROOT}/build_ios/install -x ${PROJ_ROOT}/ios/TestApp/TestApp.xcodeproj -p ${IOS_PLATFORM} -c ${PROFILE} -t ${IOS_DEV_TEAM_ID}
            else
              ruby ${PROJ_ROOT}/scripts/xcode_build.rb -i ${PROJ_ROOT}/build_ios/install -x ${PROJ_ROOT}/ios/TestApp/TestApp.xcodeproj -p ${IOS_PLATFORM}
            fi
            if ! [ "$?" -eq "0" ]; then
              echo 'xcodebuild failed!'
              exit 1
            fi
      - run:
          name: Run Simulator Tests
          no_output_timeout: "2h"
          command: |
            set -e
            if [ ${IOS_PLATFORM} != "SIMULATOR" ]; then
              echo "not SIMULATOR build, skip it."
              exit 0
            fi
            WORKSPACE=/Users/distiller/workspace
            PROJ_ROOT=/Users/distiller/project
            source ~/anaconda/bin/activate
            pip install torch torchvision --progress-bar off
            #run unit test
            cd ${PROJ_ROOT}/ios/TestApp/benchmark
            python trace_model.py
            ruby setup.rb
            cd ${PROJ_ROOT}/ios/TestApp
            instruments -s -devices
            fastlane scan
  pytorch_linux_bazel_build:
    <<: *pytorch_params
    machine:
      image: ubuntu-1604:201903-01
    steps:
    - checkout
    - setup_linux_system_environment
    - setup_ci_environment
    - run:
        name: Bazel Build
        no_output_timeout: "1h"
        command: |
          set -e
          # Pull Docker image and run build
          echo "DOCKER_IMAGE: "${DOCKER_IMAGE}
          time docker pull ${DOCKER_IMAGE} >/dev/null
          export id=$(docker run --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -t -d -w /var/lib/jenkins ${DOCKER_IMAGE})

          echo "Do NOT merge master branch into $CIRCLE_BRANCH in environment $BUILD_ENVIRONMENT"

          git submodule sync && git submodule update -q --init --recursive

          docker cp /home/circleci/project/. $id:/var/lib/jenkins/workspace

          export COMMAND='((echo "export BUILD_ENVIRONMENT=${BUILD_ENVIRONMENT}" && echo "source ./workspace/env" && echo "sudo chown -R jenkins workspace && cd workspace && .jenkins/pytorch/build.sh") | docker exec -u jenkins -i "$id" bash) 2>&1'

          echo ${COMMAND} > ./command.sh && unbuffer bash ./command.sh | ts

          # Push intermediate Docker image for next phase to use
          if [ -z "${BUILD_ONLY}" ]; then
            # Augment our output image name with bazel to avoid collisions
            output_image=${DOCKER_IMAGE}-bazel-${CIRCLE_SHA1}
            export COMMIT_DOCKER_IMAGE=$output_image
            docker commit "$id" ${COMMIT_DOCKER_IMAGE}
            time docker push ${COMMIT_DOCKER_IMAGE}
          fi

  pytorch_linux_bazel_test:
    <<: *pytorch_params
    machine:
      image: ubuntu-1604:201903-01
    steps:
    - checkout
    - setup_linux_system_environment
    - setup_ci_environment
    - run:
        name: Test
        no_output_timeout: "90m"
        command: |
          set -e
          output_image=${DOCKER_IMAGE}-bazel-${CIRCLE_SHA1}
          export COMMIT_DOCKER_IMAGE=$output_image
          echo "DOCKER_IMAGE: "${COMMIT_DOCKER_IMAGE}

          time docker pull ${COMMIT_DOCKER_IMAGE} >/dev/null

          if [ -n "${USE_CUDA_DOCKER_RUNTIME}" ]; then
            export id=$(docker run --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --runtime=nvidia -t -d -w /var/lib/jenkins ${COMMIT_DOCKER_IMAGE})
          else
            export id=$(docker run --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -t -d -w /var/lib/jenkins ${COMMIT_DOCKER_IMAGE})
          fi

          retrieve_test_reports() {
            echo "retrieving test reports"
            docker cp -L $id:/var/lib/jenkins/workspace/bazel-testlogs ./ || echo 'No test reports found!'
          }
          trap "retrieve_test_reports" ERR

          if [[ ${BUILD_ENVIRONMENT} == *"multigpu"* ]]; then
            export COMMAND='((echo "export BUILD_ENVIRONMENT=${BUILD_ENVIRONMENT}" && echo "source ./workspace/env" && echo "sudo chown -R jenkins workspace && cd workspace && .jenkins/pytorch/multigpu-test.sh") | docker exec -u jenkins -i "$id" bash) 2>&1'
          else
            export COMMAND='((echo "export BUILD_ENVIRONMENT=${BUILD_ENVIRONMENT}" && echo "export CIRCLE_PULL_REQUEST=${CIRCLE_PULL_REQUEST}" && echo "source ./workspace/env" && echo "sudo chown -R jenkins workspace && cd workspace && .jenkins/pytorch/test.sh") | docker exec -u jenkins -i "$id" bash) 2>&1'
          fi
          echo ${COMMAND} > ./command.sh && unbuffer bash ./command.sh | ts

          retrieve_test_reports
          docker stats --all --no-stream
    - store_test_results:
        path: bazel-testlogs

  pytorch_doc_test:
    environment:
      BUILD_ENVIRONMENT: pytorch-doc-test
      # TODO: stop hardcoding this
      DOCKER_IMAGE: "308535385114.dkr.ecr.us-east-1.amazonaws.com/pytorch/pytorch-linux-xenial-py3.6-gcc5.4:8bdba785b1eac4d297d5f5930f979518012a56e0"
    resource_class: medium
    machine:
      image: ubuntu-1604:201903-01
    steps:
    - checkout
    - setup_linux_system_environment
    - setup_ci_environment
    - run:
        name: Doc test
        no_output_timeout: "30m"
        command: |
          set -ex
          export COMMIT_DOCKER_IMAGE=${DOCKER_IMAGE}-${CIRCLE_SHA1}
          echo "DOCKER_IMAGE: "${COMMIT_DOCKER_IMAGE}
          time docker pull ${COMMIT_DOCKER_IMAGE} >/dev/null
          export id=$(docker run --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -t -d -w /var/lib/jenkins ${COMMIT_DOCKER_IMAGE})
          export COMMAND='((echo "export BUILD_ENVIRONMENT=${BUILD_ENVIRONMENT}" && echo "source ./workspace/env" && echo "sudo chown -R jenkins workspace && cd workspace && . ./.jenkins/pytorch/docs-test.sh") | docker exec -u jenkins -i "$id" bash) 2>&1'
          echo ${COMMAND} > ./command.sh && unbuffer bash ./command.sh | ts
