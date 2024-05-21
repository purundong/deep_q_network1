#include "deep_q_network.h"
#include <QtWidgets/QApplication>
int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
	deep_q_network w;
	w.show();
    return a.exec();
}
