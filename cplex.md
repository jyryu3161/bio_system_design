# IBM CPLEX Solver 설치 가이드

## 1. IBM CPLEX Solver 다운로드
먼저 IBM CPLEX solver를 다운로드합니다.

## 2. 설치 파일 실행 권한 설정
다운로드한 설치 파일에 실행 권한을 부여합니다.

```bash
chmod +x cplex_studio2211.linux-x86-64.bin
```

## 3. CPLEX 설치
설치 파일을 실행하여 CPLEX를 설치합니다.

```bash
./cplex_studio2211.linux_x86_64.bin
```

설치 과정 중 solver 설치 경로를 설정해야 합니다. 예를 들어 `/home/biosys/solver` 경로를 사용할 경우, 먼저 해당 폴더를 생성합니다.

```bash
cd /home/biosys/
mkdir solver
```

## 4. Pixi 환경 활성화
Pixi 환경을 활성화합니다.

```bash
pixi shell
```

<img width="1159" height="169" alt="image" src="https://github.com/user-attachments/assets/9f73c4fd-820c-40d2-97a2-cef95a829458" />

## 5. CPLEX Python 모듈 경로로 이동
CPLEX 설치 경로 내의 Python 모듈 디렉토리로 이동합니다.

```bash
cd /home/biosys/solver/cplex/python/3.10/x86-64_linux
```

> **참고**: solver 설치 경로가 `/home/biosys/solver`가 아닌 경우, 해당 부분을 실제 설치 경로로 변경하세요.

## 6. CPLEX Python 모듈 설치
Python setup 스크립트를 실행하여 CPLEX Python 모듈을 설치합니다.

```bash
pixi run python setup.py install
```

## 7. CPLEX 설치 테스트
Python을 실행하고 CPLEX 모듈을 import하여 설치를 확인합니다.

```python
python
>>> import cplex
```

<img width="1220" height="155" alt="image" src="https://github.com/user-attachments/assets/37655d61-d289-475c-9f2c-f683b6ee7aca" />

에러 없이 import가 성공하면 설치가 정상적으로 완료된 것입니다.
