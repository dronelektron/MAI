#ifndef MAP_H
#define MAP_H

#include <exception>
#include <new>

namespace NDS
{
	template<class TKey, class TVal>
	class TMap
	{
	public:
		struct TData
		{
			TKey key;
			TVal val;
		};

		struct TNode
		{
			char color;
			TData data;
			TNode* parent;
			TNode* left;
			TNode* right;
		};

		class TIterator
		{
		public:
			TIterator();
			TIterator(TNode* root, TNode* sent);

			TIterator& operator ++ ();
			bool operator != (const TIterator& it) const;
			TData& operator * ();
			TData* operator -> ();

		private:
			TNode* mSent;
			TNode* mCur;
		};

		TMap();
		~TMap();

		void Insert(const TData& data);
		TNode* Find(const TKey& key);

		TIterator Begin();
		TIterator End();

	private:
		TNode* mRoot;
		TNode* mSent;

		void RotateLeft(TNode* node);
		void RotateRight(TNode* node);
		void InsertFix(TNode* node);
		void Destroy(TNode* node);
	};
}

template<class TKey, class TVal>
NDS::TMap<TKey, TVal>::TMap()
{
	try
	{
		mSent = new TNode;
	}
	catch (const std::bad_alloc& e)
	{
		printf("No memory\n");

		std::exit(0);
	}

	mSent->color = 'b';
	mSent->parent = mSent;
	mSent->left = mSent;
	mSent->right = mSent;
	mRoot = mSent;
}

template<class TKey, class TVal>
NDS::TMap<TKey, TVal>::~TMap()
{
	Destroy(mRoot);

	delete mSent;
}

template<class TKey, class TVal>
void NDS::TMap<TKey, TVal>::RotateLeft(TNode* node)
{
	TNode* y = node->right;

	node->right = y->left;

	if (y->left != mSent)
	{
		y->left->parent = node;
	}

	y->parent = node->parent;

	if (node->parent != mSent)
	{
		if (node->parent->left == node)
		{
			node->parent->left = y;
		}
		else
		{
			node->parent->right = y;
		}
	}
	else
	{
		mRoot = y;
	}

	y->left = node;
	node->parent = y;
}

template<class TKey, class TVal>
void NDS::TMap<TKey, TVal>::RotateRight(TNode* node)
{
	TNode* y = node->left;

	node->left = y->right;

	if (y->right != mSent)
	{
		y->right->parent = node;
	}

	y->parent = node->parent;

	if (node->parent != mSent)
	{
		if (node->parent->right == node)
		{
			node->parent->right = y;
		}
		else
		{
			node->parent->left = y;
		}
	}
	else
	{
		mRoot = y;
	}

	y->right = node;
	node->parent = y;
}

template<class TKey, class TVal>
void NDS::TMap<TKey, TVal>::Insert(const TData& data)
{
	TNode* node = mRoot;
	TNode* parent = mSent;

	while (node != mSent)
	{
		parent = node;

		if (data.key < node->data.key)
		{
			node = node->left;
		}
		else if (node->data.key < data.key)
		{
			node = node->right;
		}
		else
		{
			node->data.val = data.val;
			
			return;
		}
	}

	try
	{
		node = new TNode;
	}
	catch (const std::bad_alloc& e)
	{
		printf("No memory\n");

		std::exit(0);
	}

	node->color = 'r';
	node->data = data;
	node->parent = parent;
	node->left = mSent;
	node->right = mSent;

	if (parent != mSent)
	{
		if (data.key < parent->data.key)
		{
			parent->left = node;
		}
		else
		{
			parent->right = node;
		}
	}
	else
	{
		mRoot = node;
	}

	InsertFix(node);
}

template<class TKey, class TVal>
typename NDS::TMap<TKey, TVal>::TNode* NDS::TMap<TKey, TVal>::Find(const TKey& key)
{
	TNode* node = mRoot;

	while (node != mSent)
	{
		if (key < node->data.key)
		{
			node = node->left;
		}
		else if (node->data.key < key)
		{
			node = node->right;
		}
		else
		{
			return node;
		}
	}

	return NULL;
}

template<class TKey, class TVal>
void NDS::TMap<TKey, TVal>::InsertFix(TNode* node)
{
	while (node != mRoot && node->parent->color == 'r')
	{
		if (node->parent->parent->left == node->parent)
		{
			TNode* uncle = node->parent->parent->right;

			if (uncle->color == 'r')
			{
				node->parent->parent->color = 'r';
				node->parent->color = 'b';
				uncle->color = 'b';
				node = node->parent->parent;
			}
			else
			{
				if (node->parent->right == node)
				{
					node = node->parent;

					RotateLeft(node);
				}

				node->parent->color = 'b';
				node->parent->parent->color = 'r';
				node = node->parent->parent;

				RotateRight(node);
			}
		}
		else
		{
			TNode* uncle = node->parent->parent->left;

			if (uncle->color == 'r')
			{
				node->parent->parent->color = 'r';
				node->parent->color = 'b';
				uncle->color = 'b';
				node = node->parent->parent;
			}
			else
			{
				if (node->parent->left == node)
				{
					node = node->parent;

					RotateRight(node);
				}

				node->parent->color = 'b';
				node->parent->parent->color = 'r';
				node = node->parent->parent;
				
				RotateLeft(node);
			}
		}
	}

	mRoot->color = 'b';
}

template<class TKey, class TVal>
void NDS::TMap<TKey, TVal>::Destroy(TNode* node)
{
	if (node == mSent)
	{
		return;
	}

	Destroy(node->left);
	Destroy(node->right);

	delete node;
}

template<class TKey, class TVal>
typename NDS::TMap<TKey, TVal>::TIterator NDS::TMap<TKey, TVal>::Begin()
{
	return TIterator(mRoot, mSent);
}

template<class TKey, class TVal>
typename NDS::TMap<TKey, TVal>::TIterator NDS::TMap<TKey, TVal>::End()
{
	return TIterator(mSent, mSent);
}

template<class TKey, class TVal>
NDS::TMap<TKey, TVal>::TIterator::TIterator()
{
	mSent = NULL;
	mCur = NULL;
}

template<class TKey, class TVal>
NDS::TMap<TKey, TVal>::TIterator::TIterator(TNode* root, TNode* sent)
{
	mCur = root;
	mSent = sent;

	while (mCur->left != mSent)
	{
		mCur = mCur->left;
	}
}

template<class TKey, class TVal>
typename NDS::TMap<TKey, TVal>::TIterator& NDS::TMap<TKey, TVal>::TIterator::operator ++ ()
{
	if (mCur->right != mSent)
	{
		mCur = mCur->right;

		while (mCur->left != mSent)
		{
			mCur = mCur->left;
		}

		return *this;
	}

	TNode* y = mCur->parent;

	while (y != mSent && mCur == y->right)
	{
		mCur = y;
		y = y->parent;
	}

	mCur = y;

	return *this;
}

template<class TKey, class TVal>
bool NDS::TMap<TKey, TVal>::TIterator::operator != (const TIterator& it) const
{
	return mCur != it.mCur;
}

template<class TKey, class TVal>
typename NDS::TMap<TKey, TVal>::TData& NDS::TMap<TKey, TVal>::TIterator::operator * ()
{
	return mCur->data;
}

template<class TKey, class TVal>
typename NDS::TMap<TKey, TVal>::TData* NDS::TMap<TKey, TVal>::TIterator::operator -> ()
{
	return &mCur->data;
}

#endif
