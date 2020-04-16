/*
   By Leevon Levasseur April 16, 2020
*/

#include "trace.hpp"
#include <iostream>
#include <algorithm>

// ******* Function Member Implementation *******

// ***** World function members *****
Colour World::max_to_one(Colour const& c) const
{
	float max_value = std::max(c.r, std::max(c.g, c.b));

	if (max_value > 1.0)
	{
		return (c / max_value);
	}
	else
	{
		return (c);
	}
}



// ***** Tracer function members *****
Tracer::Tracer(std::shared_ptr<World> worldPtr) : 
	mWorld_ptr{ worldPtr }
{}


Colour Tracer::trace_ray([[maybe_unused]] atlas::math::Ray<atlas::math::Vector> const& ray) const
{
	return { 0, 0, 0 };

}

Colour Tracer::trace_ray([[maybe_unused]] atlas::math::Ray<atlas::math::Vector> const& ray, 
	[[maybe_unused]] const int depth) const 
{
	return { 0, 0, 0 };
}

// ***** Shape function members *****
Shape::Shape() : mColour{ 0, 0, 0 }
{}

void Shape::setColour(Colour const& col)
{
	mColour = col;
}

Colour Shape::getColour() const
{
	return mColour;
}

void Shape::setMaterial(std::shared_ptr<Material> const& material)
{
	mMaterial = material;
}

std::shared_ptr<Material> Shape::getMaterial() const
{
	return mMaterial;
}

// ***** Camera function members *****
Camera::Camera() :
	mEye{ 0.0f, 0.0f, 500.0f },
	mLookAt{ 0.0f, 0.0f, -600.0f },
	mUp{ 0.0f, 1.0f, 0.0f },
	mU{ 1.0f, 0.0f, 0.0f },
	mV{ 0.0f, 1.0f, 0.0f },
	mW{ 0.0f, 0.0f, 1.0f }
{}

void Camera::setEye(atlas::math::Point const& eye)
{
	mEye = eye;
}

void Camera::setLookAt(atlas::math::Point const& lookAt)
{
	mLookAt = lookAt;
}

void Camera::setUpVector(atlas::math::Vector const& up)
{
	mUp = up;
}

void Camera::computeUVW()
{
	mW = glm::normalize(mEye - mLookAt);
	mU = glm::normalize(glm::cross(mUp, mW));
	mV = glm::cross(mW, mU);

	if (areEqual(mEye.x, mLookAt.x) && areEqual(mEye.z, mLookAt.z))
	{
		if (mEye.y > mLookAt.y)
		{
			mU = { 0.0f, 0.0f, 1.0f };
			mV = { 1.0f, 0.0f, 0.0f };
			mW = { 0.0f, 1.0f, 0.0f };
		}


		if (mEye.y < mLookAt.y)
		{
			mU = { 1.0f, 0.0f, 0.0f };
			mV = { 0.0f, 0.0f, 1.0f };
			mW = { 0.0f, -1.0f, 0.0f };
		}
	}
}

// ***** Sampler function members *****
Sampler::Sampler(int numSamples, int numSets) :
	mNumSamples{ numSamples }, mNumSets{ numSets }, mCount{ 0 }, mJump{ 0 }
{
	mSamples.reserve(mNumSets* mNumSamples);
	setupShuffledIndeces();
}

int Sampler::getNumSamples() const
{
	return mNumSamples;
}

void Sampler::setupShuffledIndeces()
{
	mShuffledIndeces.reserve(mNumSamples * mNumSets);
	std::vector<int> indices;

	std::random_device d;
	std::mt19937 generator(d());

	for (int j = 0; j < mNumSamples; ++j)
	{
		indices.push_back(j);
	}

	for (int p = 0; p < mNumSets; ++p)
	{
		std::shuffle(indices.begin(), indices.end(), generator);

		for (int j = 0; j < mNumSamples; ++j)
		{
			mShuffledIndeces.push_back(indices[j]);
		}
	}
}

atlas::math::Point Sampler::sampleUnitSquare()
{
	if (mCount % mNumSamples == 0)
	{
		atlas::math::Random<int> engine;
		mJump = (engine.getRandomMax() % mNumSets) * mNumSamples;
	}

	return mSamples[mJump + mShuffledIndeces[mJump + mCount++ % mNumSamples]];
}

// ***** Light function members *****
Colour Light::L([[maybe_unused]] ShadeRec& sr)
{
	return mRadiance * mColour;
}

void Light::scaleRadiance(float b)
{
	mRadiance = b;
}

void Light::setColour(Colour const& c)
{
	mColour = c;
}


// ***** Whitted function members *****

Whitted::Whitted(std::shared_ptr<World> world_ptr) : Tracer { world_ptr }
{}

Whitted::~Whitted()
{}

Colour Whitted::trace_ray(atlas::math::Ray<atlas::math::Vector> const& ray, int depth) const
{
	if (depth > mWorld_ptr->max_depth)
		return { 0, 0, 0 };
	else
	{
		ShadeRec trace_data{};
		trace_data.world = mWorld_ptr;
		trace_data.t = std::numeric_limits<float>::max();
		bool hit{};

		for (auto obj : mWorld_ptr->scene)
		{
			hit |= obj->hit(ray, trace_data);
		}

		if (hit)
		{
			trace_data.depth = depth;
			trace_data.ray = ray;
			return (trace_data.material->shade(trace_data));
		}
		else return (mWorld_ptr->background);
	}
}

// ***** Sphere function members *****
Sphere::Sphere(atlas::math::Point center, float radius) :
	mCentre{ center }, mRadius{ radius }, mRadiusSqr{ radius * radius }
{}

bool Sphere::hit(atlas::math::Ray<atlas::math::Vector> const& ray,
	ShadeRec& sr) const
{
	atlas::math::Vector tmp = ray.o - mCentre;
	float t{ std::numeric_limits<float>::max() };
	bool intersect{ intersectRay(ray, t) };

	// update ShadeRec info about new closest hit
	if (intersect && t < sr.t)
	{
		sr.normal = (tmp + t * ray.d) / mRadius;
		sr.ray = ray;
		sr.colour = mColour;
		sr.t = t;
		sr.material = mMaterial;
	}

	return intersect;
}

bool Sphere::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
	float& tMin) const
{
	const auto tmp{ ray.o - mCentre };
	const auto a{ glm::dot(ray.d, ray.d) };
	const auto b{ 2.0f * glm::dot(ray.d, tmp) };
	const auto c{ glm::dot(tmp, tmp) - mRadiusSqr };
	const auto disc{ (b * b) - (4.0f * a * c) };

	if (atlas::core::geq(disc, 0.0f))
	{
		const float kEpsilon{ 0.01f };
		const float e{ std::sqrt(disc) };
		const float denom{ 2.0f * a };

		// Look at the negative root first
		float t = (-b - e) / denom;
		if (atlas::core::geq(t, kEpsilon))
		{
			tMin = t;
			return true;
		}

		// Now the positive root
		t = (-b + e) / denom;
		if (atlas::core::geq(t, kEpsilon))
		{
			tMin = t;
			return true;
		}
	}

	return false;
}

bool Sphere::shadowHit(atlas::math::Ray<atlas::math::Vector> const& ray,
	float& tMin) const
{
	const auto tmp{ ray.o - mCentre };
	const auto a{ glm::dot(ray.d, ray.d) };
	const auto b{ 2.0f * glm::dot(ray.d, tmp) };
	const auto c{ glm::dot(tmp, tmp) - mRadiusSqr };
	const auto disc{ (b * b) - (4.0f * a * c) };

	//std::cout << "Disc = {" << disc << "} ";
	if (atlas::core::geq(disc, 0.0f))
	{
		const float kEpsilon{ 0.01f };
		const float e{ std::sqrt(disc) };
		const float denom{ 2.0f * a };

		// Look at the negative root first
		float t = (-b - e) / denom;
		if (atlas::core::geq(t, kEpsilon))
		{
			//std::cout << "Here ";
			tMin = t;
			return true;
		}

		// Now the positive root
		t = (-b + e) / denom;
		if (atlas::core::geq(t, kEpsilon))
		{
			tMin = t;
			return true;
		}
	}

	return false;
}


// ***** Triangle function members ******

Triangle::Triangle(atlas::math::Vector v0, atlas::math::Vector v1, atlas::math::Vector v2) :
	mV0{ v0 }, mV1{ v1 }, mV2{ v2 }
{}

bool Triangle::hit(atlas::math::Ray<atlas::math::Vector> const& ray,
	ShadeRec& sr) const
{
	float t{ std::numeric_limits<float>::max() };
	atlas::math::Vector v0v1{ mV1 - mV0 };
	atlas::math::Vector v0v2{ mV2 - mV0 };
	atlas::math::Vector N{ glm::cross(v0v2, v0v1) };

	float NdotDir = glm::dot(N, -ray.d);

	//std::cout << "Normal {" << N.x << ", " << N.y << ", " << N.z << "} ";

	bool intersect{ intersectRay(ray, t) };

	if (intersect && t < sr.t)
	{
		if (NdotDir > 0.0)
		{
			sr.normal = glm::normalize(N);
		}
		else
		{
			sr.normal = glm::normalize(-N);
		}
		//std::cout << "Normal {" << sr.normal.x << ", " << sr.normal.y << ", " << sr.normal.z << "} ";
		sr.ray = ray;
		sr.colour = mColour;
		sr.t = t;
		sr.material = mMaterial;
	}

	return intersect;
}

bool Triangle::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
	float& tMin) const
{
	const float kEpsilon{ 0.01f };
	atlas::math::Vector v1v0{ mV0 - mV1 };
	atlas::math::Vector v2v0{ mV0 - mV2 };
	atlas::math::Vector Eyev0{ mV0 - ray.o };


	float m = glm::cross(v2v0, ray.d).x, n = glm::cross(Eyev0, ray.d).x, p = glm::cross(v2v0, Eyev0).x;
	float q = glm::cross(ray.d, v1v0).x, s = glm::cross(v1v0, v2v0).x;

	float e1 = Eyev0.x * m - v2v0.x * n - ray.d.x * p;
	float beta = e1 / (v1v0.x * m + v2v0.x * q + ray.d.x * s);

	if (beta < 0.0)
		return false;

	float r = glm::cross(v1v0, Eyev0).x;
	float e2 = v1v0.x * n + Eyev0.x * q + ray.d.x * r;
	float gamma = e2 / (v1v0.x * m + v2v0.x * q + ray.d.x * s);

	if (gamma < 0.0)
		return false;

	if (beta + gamma > 1.0)
		return false;

	float e3 = v1v0.x * p - v2v0.x * r + Eyev0.x * s;
	float t = e3 / (v1v0.x * m + v2v0.x * q + ray.d.x * s);

	if (t < kEpsilon)
		return false;

	tMin = t;

	return true;
}

bool Triangle::shadowHit(atlas::math::Ray<atlas::math::Vector> const& ray,
	float& tMin) const
{
	const float kEpsilon{ 0.01f };
	atlas::math::Vector v1v0{ mV0 - mV1 };
	atlas::math::Vector v2v0{ mV0 - mV2 };
	atlas::math::Vector Eyev0{ mV0 - ray.o };


	float m = glm::cross(v2v0, ray.d).x, n = glm::cross(Eyev0, ray.d).x, p = glm::cross(v2v0, Eyev0).x;
	float q = glm::cross(ray.d, v1v0).x, s = glm::cross(v1v0, v2v0).x;

	float e1 = Eyev0.x * m - v2v0.x * n - ray.d.x * p;
	float beta = e1 / (v1v0.x * m + v2v0.x * q + ray.d.x * s);

	if (beta < 0.0)
		return (false);

	float r = glm::cross(v1v0, Eyev0).x;
	float e2 = v1v0.x * n + Eyev0.x * q + ray.d.x * r;
	float gamma = e2 / (v1v0.x * m + v2v0.x * q + ray.d.x * s);

	if (gamma < 0.0)
		return (false);

	if (beta + gamma > 1.0)
		return (false);

	float e3 = v1v0.x * p - v2v0.x * r + Eyev0.x * s;
	float t = e3 / (v1v0.x * m + v2v0.x * q + ray.d.x * s);

	if (t < kEpsilon)
		return (false);

	tMin = t;

	return (true);
}


// ***** Plane function members *****

Plane::Plane(atlas::math::Point p, atlas::math::Vector normal) :
	mP{ p }, mNormal{ normal }
{}

bool Plane::hit(atlas::math::Ray<atlas::math::Vector> const& ray,
	ShadeRec& sr) const
{
	float t{ std::numeric_limits<float>::max() };
	float NdotDir = glm::dot(mNormal, -ray.d);

	bool intersect{ intersectRay(ray, t) };

	if (intersect && t < sr.t)
	{
		if (NdotDir > 0.0)
		{
			sr.normal = glm::normalize(mNormal);
		}
		else
		{
			sr.normal = glm::normalize(-mNormal);
		}
		//std::cout << "Normal {" << sr.normal.x << ", " << sr.normal.y << ", " << sr.normal.z << "} ";
		sr.ray = ray;
		sr.colour = mColour;
		sr.t = t;
		sr.material = mMaterial;
	}

	return intersect;
}

bool Plane::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
	float& tMin) const
{
	const float kEpsilon{ 0.01f };
	float numerator{ glm::dot((mP - ray.o), mNormal) };
	float denom{ glm::dot(ray.d, mNormal) };

	float tmp_t{ numerator / denom };

	if (tmp_t < kEpsilon)
	{
		return false;
	}
	if (tmp_t < tMin)
	{
		tMin = tmp_t;
	}
	return true;
}

bool::Plane::shadowHit(atlas::math::Ray<atlas::math::Vector> const& ray,
	float& tMin) const
{
	const float kEpsilon{ 0.01f };
	float numerator{ glm::dot((mP - ray.o), mNormal) };
	float denom{ glm::dot(ray.d, mNormal) };

	float tmp_t{ numerator / denom };

	//std::cout << "{tmp_t = " << tmp_t << "} ";

	if (tmp_t > kEpsilon)
	{
		//std::cout << "{tmp_t = " << tmp_t << "} ";
		tMin = tmp_t;
		return true;
	}
	else
	{
		return false;
	}
}


// ***** Pinhole function members *****
Pinhole::Pinhole() : Camera{}, mDist{ 500.0f }, mZoom{ 1.0f }
{}

void Pinhole::setDistance(float distance)
{
	mDist = distance;
}

void Pinhole::setZoom(float zoom)
{
	mZoom = zoom;
}

atlas::math::Vector Pinhole::rayDirection(atlas::math::Point const& p) const
{
	const auto dir = p.x * mU + p.y * mV - mDist * mW;
	return glm::normalize(dir);
}

void Pinhole::renderScene(std::shared_ptr<World>& world) const
{
	using atlas::math::Point;
	using atlas::math::Ray;
	using atlas::math::Vector;

	Point samplePoint{}, pixelPoint{};
	Ray<atlas::math::Vector> ray{};

	ray.o = mEye;
	float avg{ 1.0f / world->sampler->getNumSamples() };

	for (int r{ 0 }; r < world->height; ++r)
	{
		for (int c{ 0 }; c < world->width; ++c)
		{
			Colour pixelAverage{ 0, 0, 0 };

			for (int j = 0; j < world->sampler->getNumSamples(); ++j)
			{
				ShadeRec trace_data{};
				trace_data.world = world;
				trace_data.t = std::numeric_limits<float>::max();
				samplePoint = world->sampler->sampleUnitSquare();
				pixelPoint.x = c - 0.5f * world->width + samplePoint.x;
				pixelPoint.y = r - 0.5f * world->height + samplePoint.y;
				ray.d = rayDirection(pixelPoint);
				bool hit{};

				for (auto obj : world->scene)
				{
					hit |= obj->hit(ray, trace_data);
				}

				if (hit)
				{
					trace_data.hitPoint = ray.o + trace_data.t * ray.d;
					pixelAverage += trace_data.material->shade(trace_data);
				}
			}

			pixelAverage.r *= avg;
			pixelAverage.g *= avg;
			pixelAverage.b *= avg;
			world->image.push_back(world->max_to_one(pixelAverage));
		}
	}
}


// ***** Regular function members *****
Regular::Regular(int numSamples, int numSets) : Sampler{ numSamples, numSets }
{
	generateSamples();
}

void Regular::generateSamples()
{
	int n = static_cast<int>(glm::sqrt(static_cast<float>(mNumSamples)));

	for (int j = 0; j < mNumSets; ++j)
	{
		for (int p = 0; p < n; ++p)
		{
			for (int q = 0; q < n; ++q)
			{
				mSamples.push_back(
					atlas::math::Point{ (q + 0.5f) / n, (p + 0.5f) / n, 0.0f });
			}
		}
	}
}

// ***** Regular function members *****
Random::Random(int numSamples, int numSets) : Sampler{ numSamples, numSets }
{
	generateSamples();
}

void Random::generateSamples()
{
	atlas::math::Random<float> engine;
	for (int p = 0; p < mNumSets; ++p)
	{
		for (int q = 0; q < mNumSamples; ++q)
		{
			mSamples.push_back(atlas::math::Point{
				engine.getRandomOne(), engine.getRandomOne(), 0.0f });
		}
	}
}

Multijittered::Multijittered(int numSamples, int numSets) : Sampler{ numSamples, numSets }
{
	generateSamples();
}

void Multijittered::generateSamples()
{
	int n = static_cast<int>(glm::sqrt(static_cast<float>(mNumSamples)));
	float subcell_width = 1.0f / ((float)mNumSamples);
	atlas::math::Random<float> engine;

	//fill the samples array with dummy points to allow us to use the [ ] notation when we set the
	// initial patterns

	atlas::math::Point fill_point;
	for (int j = 0; j < mNumSamples * mNumSets; ++j)
		mSamples.push_back(fill_point);

	//Distribute points in the initial patterns

	for (int p = 0; p < mNumSets; ++p)
	{
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < n; ++j)
			{
				mSamples[i * n + j + p * mNumSamples].x = (i * n + j) * subcell_width + engine.getRandomOne() * subcell_width;
				mSamples[i * n + j + p * mNumSamples].y = (j * n + i) * subcell_width + engine.getRandomOne() * subcell_width;
			}
		}
	}

	//Shuffle x coordinates

	for (int p = 0; p < mNumSets; ++p)
	{
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < n; ++j)
			{
				int k = j + (rand() % static_cast<int>(n - j));
				float t = mSamples[i * n + j + p * mNumSamples].x;
				mSamples[i * n + j + p * mNumSamples].x = mSamples[i * n + k + p * mNumSamples].x;
				mSamples[i * n + k + p * mNumSamples].x = t;
			}
		}
	}

	//Shuffle y coordinates

	for (int p = 0; p < mNumSets; ++p)
	{
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < n; ++j)
			{
				int k = j + (rand() % static_cast<int>(n - j));
				float t = mSamples[j * n + i + p * mNumSamples].y;
				mSamples[j * n + i + p * mNumSamples].y = mSamples[k * n + i + p * mNumSamples].y;
				mSamples[k * n + i + p * mNumSamples].y = t;
			}
		}
	}
}

// ***** Lambertian function members *****
Lambertian::Lambertian() :
	mDiffuseColour{}, mDiffuseReflection{}
{}

Lambertian::Lambertian(Colour diffuseColor, float diffuseReflection) :
	mDiffuseColour{ diffuseColor }, mDiffuseReflection{ diffuseReflection }
{}

Colour Lambertian::fn([[maybe_unused]] ShadeRec const& sr,
	[[maybe_unused]] atlas::math::Vector const& reflected,
	[[maybe_unused]] atlas::math::Vector const& incoming) const
{
	return mDiffuseColour * mDiffuseReflection * glm::one_over_pi<float>();
}

Colour Lambertian::rho([[maybe_unused]] ShadeRec const& sr,
	[[maybe_unused]] atlas::math::Vector const& reflected) const
{
	return mDiffuseColour * mDiffuseReflection;
}

void Lambertian::setDiffuseReflection(float kd)
{
	mDiffuseReflection = kd;
}

void Lambertian::setDiffuseColour(Colour const& colour)
{
	mDiffuseColour = colour;
}

// ***** GlossySpecular function memebers *****

GlossySpecular::GlossySpecular() :
	mDiffuseColour{}, mExp{}, mDiffuseReflection{}
{}

GlossySpecular::GlossySpecular(Colour diffuseColour, float exp, float diffuseReflection) :
	mDiffuseColour{ diffuseColour }, mExp{ exp }, mDiffuseReflection{ diffuseReflection }
{}

Colour GlossySpecular::fn([[maybe_unused]] ShadeRec const& sr,
	[[maybe_unused]] atlas::math::Vector const& reflected,
	[[maybe_unused]] atlas::math::Vector const& incoming) const
{
	Colour L{ 0 };
	float ndotIncoming{ glm::dot(sr.normal, incoming) };
	atlas::math::Vector r{ -incoming + 2.0f * sr.normal * ndotIncoming };
	float rdotReflected{ glm::dot(r, reflected) };
	Colour glossyColour{ 1, 1, 1 };

	if (rdotReflected > 0.0f)
	{
		L = (mDiffuseReflection * glossyColour * pow(rdotReflected, mExp));
	}
	return L;
}

Colour GlossySpecular::rho([[maybe_unused]] ShadeRec const& sr,
	[[maybe_unused]] atlas::math::Vector const& reflected) const
{
	return { 0, 0, 0 };
}

void GlossySpecular::setExponent(float exp)
{
	mExp = exp;
}

void GlossySpecular::setDiffuseReflection(float ks)
{
	mDiffuseReflection = ks;
}

void GlossySpecular::setDiffuseColour(Colour const& colour)
{
	mDiffuseColour = colour;
}

// ***** PerfectSpecular function members *****
PerfectSpecular::PerfectSpecular() :
	mCr{}, mKr{}
{}

PerfectSpecular::PerfectSpecular(float Kr, Colour Cr) :
	mKr{ Kr }, mCr{ Cr }
{}

Colour PerfectSpecular::sample_f([[maybe_unused]] ShadeRec const& sr,
	[[maybe_unused]] atlas::math::Vector const& reflected,
	[[maybe_unused]] atlas::math::Vector& incoming) const
{
	float ndotReflected = glm::dot(sr.normal, reflected);
	incoming = -reflected + 2.0f * ndotReflected * sr.normal;

	return (mKr * mCr / glm::dot(sr.normal, incoming));
}

Colour PerfectSpecular::fn([[maybe_unused]] ShadeRec const& sr,
	[[maybe_unused]] atlas::math::Vector const& reflected,
	[[maybe_unused]] atlas::math::Vector const& incoming) const
{
	return { 0, 0, 0 };
}

Colour PerfectSpecular::rho([[maybe_unused]] ShadeRec const& sr,
	[[maybe_unused]] atlas::math::Vector const& reflected) const
{
	return { 0, 0, 0 };
}

void PerfectSpecular::set_kr(float kr)
{
	mKr = kr ;
}

void PerfectSpecular::set_cr(Colour cr)
{
	mCr = cr;
}

// ***** PerfectTransmitter function members *****
PerfectTransmitter::PerfectTransmitter() :
	mKt{}, mIor{}
{}

PerfectTransmitter::PerfectTransmitter(float kt, float ior) :
	mKt{ kt }, mIor{ ior }
{}

Colour PerfectTransmitter::sample_f([[maybe_unused]] ShadeRec const& sr,
	atlas::math::Vector const& reflected,
	atlas::math::Vector& incoming) const
{
	atlas::math::Normal n{ sr.normal };
	float cos_thetai = glm::dot(n, reflected);
	float eta = mIor;

	if (cos_thetai < 0.0)
	{
		cos_thetai = -cos_thetai;
		n = -n;
		eta = 1.0f / eta;
	}

	float tmp = 1.0f - (1.0f - cos_thetai * cos_thetai) / (eta * eta);
	float cos_theta2 = sqrt(tmp);
	incoming = -reflected / eta - (cos_theta2 - cos_thetai / eta) * n;

	return (mKt / (eta * eta) * Colour{ 1, 1, 1 } / fabs(glm::dot(sr.normal, incoming)));
}

Colour PerfectTransmitter::fn([[maybe_unused]] ShadeRec const& sr,
	[[maybe_unused]] atlas::math::Vector const& reflected,
	[[maybe_unused]] atlas::math::Vector const& incoming) const
{
	return { 0, 0, 0 };
}

Colour PerfectTransmitter::rho([[maybe_unused]] ShadeRec const& sr,
	[[maybe_unused]] atlas::math::Vector const& reflected) const
{
	return { 0, 0, 0 };
}

bool PerfectTransmitter::tir([[maybe_unused]] ShadeRec const& sr) const
{
	atlas::math::Vector wo{ -sr.ray.d };
	float cos_thetai = glm::dot(sr.normal, wo);
	float eta = mIor;

	if (cos_thetai < 0.0)
		eta = 1.0f / eta;

	return (1.0f - (1.0f - cos_thetai * cos_thetai) / (eta * eta) < 0.0);
}



// ***** Matte function members *****
Matte::Matte() :
	Material{},
	mDiffuseBRDF{ std::make_shared<Lambertian>() },
	mAmbientBRDF{ std::make_shared<Lambertian>() }
{}

Matte::Matte(float kd, float ka, Colour colour) : Matte{}
{
	setDiffuseReflection(kd);
	setAmbientReflection(ka);
	setDiffuseColour(colour);
}

void Matte::setDiffuseReflection(float k)
{
	mDiffuseBRDF->setDiffuseReflection(k);
}

void Matte::setAmbientReflection(float k)
{
	mAmbientBRDF->setDiffuseReflection(k);
}

void Matte::setDiffuseColour(Colour colour)
{
	mDiffuseBRDF->setDiffuseColour(colour);
	mAmbientBRDF->setDiffuseColour(colour);
}

Colour Matte::shade(ShadeRec& sr)
{
	using atlas::math::Ray;
	using atlas::math::Vector;

	Vector wo = -sr.ray.d;
	Colour L = mAmbientBRDF->rho(sr, wo) * sr.world->ambient->L(sr);
	size_t numLights = sr.world->lights.size();

	for (size_t i{ 0 }; i < numLights; ++i)
	{
		Vector wi = sr.world->lights[i]->getDirection(sr);
		float nDotWi = glm::dot(sr.normal, wi);
		bool in_shadow = false;

		if (nDotWi > 0.0f)
		{
			Ray shadowRay{ sr.hitPoint, wi };
			in_shadow = sr.world->lights[i]->inShadow(shadowRay, sr);

			if (!in_shadow)
			{
				L += mDiffuseBRDF->fn(sr, wo, wi) * sr.world->lights[i]->L(sr) *
					nDotWi;
			}
		}
	}

	return L;
}

// ***** Phong function members *****

Phong::Phong() :
	Material{},
	mDiffuseBRDF{ std::make_shared<Lambertian>() },
	mAmbientBRDF{ std::make_shared<Lambertian>() },
	mSpecularBRDF{ std::make_shared<GlossySpecular>() }
{}

Phong::Phong(float kd, float ka, float exp, Colour colour) : Phong{}
{
	setDiffuseReflection(kd);
	setAmbientReflection(ka);
	setExponent(exp);
	setDiffuseColour(colour);
}

void Phong::setDiffuseReflection(float k)
{
	mDiffuseBRDF->setDiffuseReflection(k);
	mSpecularBRDF->setDiffuseReflection(k);
}

void Phong::setAmbientReflection(float k)
{
	mAmbientBRDF->setDiffuseReflection(k);
}


void Phong::setExponent(float exp)
{
	mSpecularBRDF->setExponent(exp);
}

void Phong::setDiffuseColour(Colour colour)
{
	mDiffuseBRDF->setDiffuseColour(colour);
	mAmbientBRDF->setDiffuseColour(colour);
	mSpecularBRDF->setDiffuseColour(colour);
}


Colour Phong::shade(ShadeRec& sr)
{
	using atlas::math::Ray;
	using atlas::math::Vector;

	Vector wo = -sr.ray.d;
	Colour L = mAmbientBRDF->rho(sr, wo) * sr.world->ambient->L(sr);
	size_t numLights = sr.world->lights.size();
	bool in_shadow = false;

	for (size_t i{ 0 }; i < numLights; ++i)
	{
		Vector wi = sr.world->lights[i]->getDirection(sr);
		float nDotWi = glm::dot(sr.normal, wi);

		if (nDotWi > 0.0f)
		{
			Ray shadowRay{ sr.hitPoint, wi };
			in_shadow = sr.world->lights[i]->inShadow(shadowRay, sr);

			if (!in_shadow)
			{
				L += (mDiffuseBRDF->fn(sr, wo, wi) + mSpecularBRDF->fn(sr, wo, wi)) *
					sr.world->lights[i]->L(sr) * nDotWi;
			}
		}
	}

	return L;
}


// ***** Reflective function members *****
Reflective::Reflective() : Phong(), mReflectiveBRDF{ std::make_shared<PerfectSpecular>() }
{}

Reflective::Reflective(float kd, float ka, float exp, Colour colour, float kr, Colour cr) :
	Phong(kd, ka, exp, colour), mReflectiveBRDF { std::make_shared<PerfectSpecular>(kr, cr) }
{}

Colour Reflective::shade([[maybe_unused]] ShadeRec& sr)
{
	using atlas::math::Ray;
	using atlas::math::Vector;

	Colour L(Phong::shade(sr)); //direct Illumination

	Vector w0 = -sr.ray.d;
	Vector wi;

	Colour fr = mReflectiveBRDF->sample_f(sr, w0, wi);
	Ray reflected_ray{ sr.hitPoint, wi };

	L += fr * sr.world->tracer_ptr->trace_ray(reflected_ray, sr.depth + 1) *
		glm::dot(sr.normal, wi);

	return L;
}

// ***** Transparent function members *****
Transparent::Transparent() : Phong(), 
	mReflectiveBRDF{ std::make_shared<PerfectSpecular>() },
	mSpecularBTDF{ std::make_shared<PerfectTransmitter>() }
{}

Transparent::Transparent(float kd, float ka, float exp, Colour colour, float kr, Colour cr, float kt, float ior) :
	Phong(kd, ka, exp, colour), mReflectiveBRDF{ std::make_shared<PerfectSpecular>(kr, cr) },
	mSpecularBTDF{ std::make_shared<PerfectTransmitter>(kt, ior) }
{}

Colour Transparent::shade([[maybe_unused]] ShadeRec& sr)
{
	using atlas::math::Vector;
	using atlas::math::Ray;

	Colour L{ Phong::shade(sr) };

	Vector wo = -sr.ray.d;
	Vector wi;
	Colour fr = mReflectiveBRDF->sample_f(sr, wo, wi);
	Ray reflected_ray{ sr.hitPoint, wi };

	if (mSpecularBTDF->tir(sr))
	{
		L += sr.world->tracer_ptr->trace_ray(reflected_ray, sr.depth + 1);
		// kr = 1.0;
	}
	else
	{
		Vector wt;
		Colour ft = mSpecularBTDF->sample_f(sr, wo, wt); //computes wt
		Ray transmitted_ray{ sr.hitPoint, wt };

		L += fr * sr.world->tracer_ptr->trace_ray(reflected_ray, sr.depth + 1)
			* fabs(glm::dot(sr.normal, wi));
		L += ft * sr.world->tracer_ptr->trace_ray(transmitted_ray, sr.depth + 1)
			* fabs(glm::dot(sr.normal, wt));
	}

	return (L);
}



// ***** Point Light function members *****

PointLight::PointLight() : Light{}
{}

PointLight::PointLight(atlas::math::Point const& origin) : Light{}
{
	setOrigin(origin);
}

void PointLight::setOrigin(atlas::math::Point const& origin)
{
	mOrigin = origin;
}

atlas::math::Point PointLight::getOrigin([[maybe_unused]] ShadeRec& sr)
{
	return mOrigin;
}

atlas::math::Vector PointLight::getDirection([[maybe_unused]] ShadeRec& sr)
{
	return glm::normalize(mOrigin - sr.hitPoint);
}

bool PointLight::inShadow(atlas::math::Ray<atlas::math::Vector> const& shadowRay,
	[[maybe_unused]] ShadeRec& sr) const
{
	float t;
	size_t numObjects = sr.world->scene.size();
	float d = sqrt(pow((mOrigin.x - shadowRay.o.x), 2)
		+ pow((mOrigin.x - shadowRay.o.x), 2)
		+ pow((mOrigin.x - shadowRay.o.x), 2));

	for (size_t i{ 0 }; i < numObjects; ++i)
	{
		if (sr.world->scene[i]->shadowHit(shadowRay, t) && (t < d))
		{
			return true;
		}
	}

	return false;
}

// ***** Directional function members *****

Directional::Directional() : Light{}
{}

Directional::Directional(atlas::math::Vector const& d) : Light{}
{
	setDirection(d);
}

void Directional::setDirection(atlas::math::Vector const& d)
{
	mDir = glm::normalize(d);
}

atlas::math::Vector Directional::getDirection([[maybe_unused]] ShadeRec& sr)
{
	return mDir;
}

bool Directional::inShadow(atlas::math::Ray<atlas::math::Vector> const& shadowRay,
	[[maybe_unused]] ShadeRec& sr) const
{
	float t;
	size_t numObjects = sr.world->scene.size();

	for (size_t i{ 0 }; i < numObjects; ++i)
	{
		if (sr.world->scene[i]->shadowHit(shadowRay, t)) //false if t<0
		{
			return true;
		}
	}

	return false;
}

// ***** Ambient function members *****

Ambient::Ambient() : Light{}
{}

atlas::math::Vector Ambient::getDirection([[maybe_unused]] ShadeRec& sr)
{
	return atlas::math::Vector{ 0.0f };
}

bool Ambient::inShadow([[maybe_unused]] atlas::math::Ray<atlas::math::Vector> const& ray,
	[[maybe_unused]] ShadeRec& sr) const
{
	return false;
}

// ******* Driver Code *******

int main()
{
	using atlas::math::Point;
	using atlas::math::Ray;
	using atlas::math::Vector;

	std::shared_ptr<World> world{ std::make_shared<World>() };

	world->width = 1000;
	world->height = 1000;
	world->background = { 0, 0, 0 };
	world->sampler = std::make_shared<Multijittered>(4, 83);
	world->max_depth = 4;
	world->tracer_ptr = std::make_shared<Whitted>(world);


	//Yellow Triangle
	world->scene.push_back(
		std::make_shared<Triangle>(atlas::math::Vector{ -192, 40, -408 }, atlas::math::Vector{ -128, 128, -424 }, atlas::math::Vector{ -64, 100, -440 }));
	world->scene[0]->setMaterial(
		std::make_shared<Matte>(0.50f, 0.05f, Colour{ 1, 1, 0 }));
	world->scene[0]->setColour({ 1, 1, 0 });

	//Red Sphere
	world->scene.push_back(
		std::make_shared<Sphere>(atlas::math::Point{ 0, 0, -600 }, 128.0f));
	world->scene[1]->setMaterial(
		std::make_shared<Reflective>(0.50f, 0.05f, 100.0f, Colour{ 1, 0, 0 }, 0.75f, Colour{ 1, 1, 1 }));
	world->scene[1]->setColour({ 1, 0, 0 });


	//Blue Sphere
	world->scene.push_back(
		std::make_shared<Sphere>(atlas::math::Point{ 192, 64, -700 }, 64.0f));
	world->scene[2]->setMaterial(
		std::make_shared<Phong>(0.50f, 0.05f, 30.0f, Colour{ 0, 0, 1 }));
	world->scene[2]->setColour({ 0, 0, 1 });


	//Green Sphere
	world->scene.push_back(
		std::make_shared<Sphere>(atlas::math::Point{ -188, 64, -900 }, 64.0f));
	world->scene[3]->setMaterial(
		std::make_shared<Matte>(0.50f, 0.05f, Colour{ 0, 1, 0 }));
	world->scene[3]->setColour({ 0, 1, 0 });


	//Ground Plane
	world->scene.push_back(
		std::make_shared<Plane>(atlas::math::Point{ 0, 126, 0 }, atlas::math::Vector{ 0,-1,0 }));
	world->scene[4]->setMaterial(
		std::make_shared<Reflective>(0.50f, 0.05f, 50.0f, Colour{ 0.30f, 0.26f, 0.0f }, 0.25f, Colour{ 1, 1, 1 }));
	world->scene[4]->setColour({ 0.30f, 0.26f, 0.0f });

	//Light Blue Sphere
	world->scene.push_back(
		std::make_shared<Sphere>(atlas::math::Point{ -150, 40, -398 }, 20.0f));
	world->scene[5]->setMaterial(
		std::make_shared<Transparent>(0.50f, 0.05f, 2000.0f, Colour{ 0.21f, 0.32f, 0.36f }, 0.9f, Colour{ 1, 1, 1 }, 0.9f, 1.5f));
	world->scene[5]->setColour({ 0.21f, 0.32f, 0.36f });

	//Sky Plane
	/*world->scene.push_back(
		std::make_shared<Plane>(atlas::math::Point{ 0, 0, -10000 }, atlas::math::Vector{ 0, 0, -1 }));
	world->scene[5]->setMaterial(
		std::make_shared<Matte>(0.50f, 0.05f, Colour{ 0.21f, 0.32f, 0.36f }));
	world->scene[5]->setColour({ 0.21f, 0.32f, 0.36f });*/



	world->ambient = std::make_shared<Ambient>();
	world->lights.push_back(
		std::make_shared<Directional>(Directional({ 0, -30, 1024 })));
	world->lights.push_back(
		std::make_shared<PointLight>(PointLight({ -1500, -800, 1024 })));
	world->lights.push_back(
		std::make_shared<PointLight>(PointLight({ -500, -2000, 500 })));

	world->ambient->setColour({ 1, 1, 1 });
	world->ambient->scaleRadiance(0.50f);

	world->lights[0]->setColour({ 1, 1, 1 });
	world->lights[0]->scaleRadiance(6.15f);

	world->lights[1]->setColour({ 1, 1, 1 });
	world->lights[1]->scaleRadiance(6.15f);

	world->lights[2]->setColour({ 1, 1, 1 });
	world->lights[2]->scaleRadiance(6.15f);

	/*world->lights[1]->setColour({ 1, 1, 1 });
	world->lights[1]->scaleRadiance(5.0f);*/

	// set up camera
	Pinhole camera{};

	// change camera position here
	camera.setEye({ 0.0f, -100.0f, 500.0f });

	camera.computeUVW();

	camera.renderScene(world);

	saveToFile("raytrace.bmp", world->width, world->height, world->image);

	return 0;
}

void saveToFile(std::string const& filename,
	std::size_t width,
	std::size_t height,
	std::vector<Colour> const& image)
{
	std::vector<unsigned char> data(image.size() * 3);

	for (std::size_t i{ 0 }, k{ 0 }; i < image.size(); ++i, k += 3)
	{
		Colour pixel = image[i];
		data[k + 0] = static_cast<unsigned char>(pixel.r * 255);
		data[k + 1] = static_cast<unsigned char>(pixel.g * 255);
		data[k + 2] = static_cast<unsigned char>(pixel.b * 255);
	}

	stbi_write_bmp(filename.c_str(),
		static_cast<int>(width),
		static_cast<int>(height),
		3,
		data.data());
}

