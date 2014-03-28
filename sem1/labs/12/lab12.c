#include <stdio.h>

int main(void)
{
	int n, num, dif, prev;

	do
	{
		printf("Введите n (выход - 0): ");
		scanf("%d", &n);
		printf("Исходное число n = %d\n", n);

		prev = -1;
		dif = 1;
		num = n;
		n = (n >= 0 ? n : -n);

		while (n)
		{
			int ost = n % 10;

			n /= 10;

			if (prev == -1)
			{
				prev = ost;

				continue;
			}

			if (ost == prev)
			{
				dif = 0;

				break;
			}
			else prev = ost;
		}

		printf("%s\n", dif ? "Не содержит одинаковых смежных разрядов" : "Содержит одинаковые смежные разряды");
	} while (num);

	return 0;
}
