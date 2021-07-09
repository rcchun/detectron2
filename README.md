Detectron2

detectron2란 FAIR(Facebook Artificial Intelligence Research)에서 만든 pytorch 기반 object detection과 sementic segemanation을 위한 training/inference 플랫폼으로 Detectron2를 이용해 학습을 하게 되면, 우리가 일반적으로 딥러닝 모델을 짤 때 구현하던 training loop를 짜지 않고 engine을 이용하여 학습 과정을 추상화 할 수 있어, 연구자 혹은 개발자는 모델 개발 자체에만 집중할 수 있다.

Detectron2가 다른 오픈 소스들에 비해 빠른이유는 python 최적화가 잘되어있기도 합니다만, 그 외에 연산량이 많이 드는 부분을 python이 아닌 CUDA와 C로 구현했기에 보다 좋은 성능을 냈습니다. box iou를 계산하는 부분이나, defromable conv 부분 등은 연산량이 많이 드는 부분인데, 이 부분을 CUDA로 구현하였다.
