Linuxの根幹だったりを色々編集できるようにするために開発された言語が「C言語」ってやつだってのはいつか話したことですけども。
それ故にC言語ではI/O出力が高水準と低水準に分かれている＆使い分けがガチで重要になっておりまして。
低水準I/O関数：プロセスの内容と1対1対応になっているもの。ファイルアクセス用のI/O関数に「()」がつくとこちらに該当する
高水準I/O関数：プロセスの内容に対して1対多の対応が可能になるもの(複数の入力を一括で扱う機能を持っている、俗にいう「ストリーム」ってやつです)。コマンドに「f」が含まれてればこちらになる

使用するライブラリによって使えるI/O関数も勿論異なるのですが、低水準は1つ、他は全て高水準に該当します。低水準I/Oにしても高水準I/Oにしても、現在実行中の1つのプログラム(これをプロセスと呼んでいます)に対して3つのI/Oのチャンネルを持っております。

なおワイの経験上、色々なプログラム例を見ながら学習した方が早かった気がするのでそうします。
```C
#include <unistd.h>#include <fcntl.h>#include <stdio.h>#include <stdlib.h>#include <string.h>#include <sys/wait.h> //やった工程全部表示する

//unixのstd(standardの略称)で、システムコールが全てこれに含まれている&これだけ低水準I/Oに該当する

#define BUFFER_SIZE 1024 //データの上限値を定義
int main() {
    char buffer[BUFFER_SIZE];
    int fd, pipefd[2]; //ファイルディスクリプタ(ファイルの背番号的なやつ)の定義
    pid_t pid; //pidを定義

    // getcwd(): 現在の作業ディレクトリを取得
    if (getcwd(buffer, BUFFER_SIZE) != NULL) {
        printf("現在のディレクトリ: %s\n", buffer); //現在のディレクトリを表示
    }[1]

    // chdir(): ディレクトリを変更
    chdir("/tmp");
    printf("ディレクトリを/tmpに変更しました\n");

    // ファイルを作成してlseek()でポジションを移動
    fd = open("test.txt", O_RDWR | O_CREAT, 0644);
    write(fd, "Hello, World!", 13);
    lseek(fd, 0, SEEK_SET);	//ファイルの読み込み位置を決定する
    read(fd, buffer, 13);
    printf("ファイルの内容: %s\n", buffer);
    close(fd);

    // pipe()を使用してパイプを作成
    if (pipe(pipefd) == -1) {
        perror("pipe");
        exit(1);
    }

    // fork()を使用して子プロセスを作成
    pid = fork(); //子プロセスの設定

    if (pid == -1) {
        perror("fork");
        exit(1);
    } else if (pid == 0) {
        // 子プロセス
        close(pipefd[1]);
        dup2(pipefd[0], STDIN_FILENO);
        close(pipefd[0]);

        printf("子プロセス PID: %d, 親プロセス PID: %d\n", getpid(), getppid());

        // exec()を使用して新しいプログラムを実行
        execlp("wc", "wc", "-c", NULL); //現在実行中のコマンドに代わり指定したコマンドを実行する
        perror("execlp");
        exit(1);
    } else {
        // 親プロセス
        close(pipefd[0]);
        write(pipefd[1], "This is a test message", 22);
        close(pipefd[1]);

        printf("親プロセス PID: %d\n", getpid());

        // sleep()を使用して1秒間停止
        sleep(1);

        // 子プロセスの終了を待つ
        wait(NULL);
    }

    return 0;
}

getpid()：通常のプロセスID(PID)の取得
getppid()：親プロセスIDの取得
chdir()：現在の位置からパス内を移動する
sleep()：プログラムを停止する。sleep(1)だと1秒間
getcwd()：絶対パスの入った文字列を返す
pipe()：パイプを生成する
dup()：FDを複製する。0なら標準入力、1なら標準出力、2ならエラー出力となる

fcntl.h：
マクロ系の設定を行う
O_RDONLY：ファイルを読み取り専用(Read Only)として開くマクロ設定する
O_WRONLY：書き込み専用(Write Only)
O_CREAT：ファイルが存在しない場合に新規作成する
O_TRUNC：ファイルん中を空にする
F_SETFL：FDのステータスフラグ(status FLag)を設定(SET)する
O_NONBLOCK：I/O処理が即座に完了せずとも制御を呼び出すモードのフラグを設定する
set_nblk：ノンブロックモードにする
lsock：ソケットを生成する
rfds：FDの集合。これを「rfds_tmp」にコピペすることで状態を維持させられる

sys/stat.h：
ファイル状態を確認する

sys/socket.h：
ソケットを扱えるようにする
socket：新たなソケットを生成する。
domain：「AF_INET(IPv4)」、「AF_INET6(IPv6)」、「AF_UNIX(unix)」
type：「SOCK_STRAM(TCP)」、「SOCK_DGRAM(UDP)」
protocol：通常は0、あとは自動選択
bind：IPアドレスやポート番号を紐づける。鯖側は「0.0.0.0」とか使う
listen：TCPの「SOCK_STRAM」で有用
accept：鯖側で新たなソケットを生成して返す
connect：蔵側で接続要求を送る
close：ソケット通信を閉じる

arpa/inet.h：
htons：16ビットでホストバイトオーダーをネットワークバイトオーダーに変換する(host to network short)。逆は「ntohs」。32ビット(long)の場合は「htonl」と「ntohl」になる
inet_ntoa：IPv4をネットワークバイトオーダーからドット区切り文字列に変換する

signal.h：
SIGCHLD：子プロセスが停止したときに親プロセスに送られるシグナル
SIG_IGN：ゾンビ化を避けるために子プロセスが停止したことを無視する

stdio.h：
ストリーム用の基本的なI/O関数を提供している
printf：出力。%dで整数、%sで文字列、%fで浮動小数点数
fflush：強制吐き出し
buff：データを一時保存するためのメモリ領域を生成する
itoa：整数(int)を文字列(char)に変換させる。逆は「atoi」

stdlib.h：
メモリ関係やプログラム制御など、「プログラミングといえばこれ」という基本的な機能を提供している
malloc：メモリを割り当てる
free：メモリを解放する。これが為されないと「メモリリーク」という安定性が損なわれる状態になる

string.h：
文字列型を提供する

stdint.h：
整数型を提供する

stdbool.h：
真偽値(True/False)を提供している

math.h：
数学関連の機能を提供している
```